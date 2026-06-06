'''
Synthetic dataset from Pegasus simulation.
_FLIGHT_LOG_COLUMNS = [
    # time 0:2
    "sim_time_s",
    "wall_time_s",               # Unix wall-clock (time.time()) — use this to merge with ekf2_log
    # --- ground-truth state (ENU, from Isaac Sim physics) 2:15 ---
    "gt_pos_x_m",   "gt_pos_y_m",   "gt_pos_z_m",          # ENU position
    "gt_vel_x_ms",  "gt_vel_y_ms",  "gt_vel_z_ms",          # ENU velocity
    "gt_qx",        "gt_qy",        "gt_qz",        "gt_qw", # attitude (FLU→ENU) xyzw
    "gt_wx_rads",   "gt_wy_rads",   "gt_wz_rads",            # body angular velocity
    # --- IMU (FRD body frame, noisy) 15:21 ---
    "imu_ax_ms2",   "imu_ay_ms2",   "imu_az_ms2",            # accelerometer
    "imu_gx_rads",  "imu_gy_rads",  "imu_gz_rads",           # gyroscope
    # --- Barometer 21:24 ---
    "baro_pressure_pa", "baro_alt_m", "baro_temp_k",
    # --- Magnetometer (body frame, noisy) 24:27 ---
    "mag_x_gauss",  "mag_y_gauss",  "mag_z_gauss",
    # --- GPS (noisy + possibly spoofed) 27:33 ---
    "gps_lat_deg",  "gps_lon_deg",  "gps_alt_m",
    "gps_vn_ms",    "gps_ve_ms",    "gps_vd_ms",
    # --- GPS ground truth (unaffected by spoofing) 33:36 ---
    "gps_lat_gt_deg", "gps_lon_gt_deg", "gps_alt_gt_m",
    # --- GPS spoofing bias (ENU metres) 36:39 ---
    "spoof_bias_x_m", "spoof_bias_y_m", "spoof_bias_z_m",
    # --- Spoofing flag 39 ---
    "spoofing_active",      # 1 = drift spoofing on, 0 = off
    # --- GPS replay attack flag 40 ---
    "gps_replay_active",    # 1 = replay attack in progress, 0 = live GPS
    # --- Gyroscope bias fault flag 41 ---
    "gyro_bias_active",     # 1 = constant bias injected into IMU gyroscope, 0 = nominal
    # --- Motor fault flag 42 ---
    "motor_fault_active",   # 1 = at least one rotor is capped by a fault, 0 = nominal
    # --- Per-rotor fault fractions 43:47 (1.0 = nominal, <1.0 = capped, 0.0 = dead) ---
    "motor_fault_r0_frac", "motor_fault_r1_frac", "motor_fault_r2_frac", "motor_fault_r3_frac",
    # --- Motor delay flag 47 ---
    "motor_delay_active",   # 1 = first-order response lag on affected rotors, 0 = nominal
    # --- GPS denial flag 48 ---
    "gps_denial_active",    # 1 = GPS state frozen (denial attack), 0 = live GPS
    # --- IMU high-frequency noise flag 49 ---
    "imu_hf_noise_active",  # 1 = extra HF noise injected on gyro+accel, 0 = nominal
    # --- Control inputs (rotor angular speeds rad/s from PX4 via HIL) 50:54 ---
    "rotor0_rads",  "rotor1_rads",  "rotor2_rads",  "rotor3_rads",
    # -----------------------------------------------------------------------
    # PX4 EKF2 estimated states  (from MAVLink GCS stream, port 14550)
    # -----------------------------------------------------------------------
    # LOCAL_POSITION_NED (#32) — EKF2 position + velocity in NED frame 54:60
    "ekf_pos_n_m",  "ekf_pos_e_m",  "ekf_pos_d_m",     # NED position (m)
    "ekf_vel_n_ms", "ekf_vel_e_ms", "ekf_vel_d_ms",     # NED velocity (m/s)
    # ATTITUDE_QUATERNION (#31) — EKF2 attitude (MAVLink: q1=w, q2=x, q3=y, q4=z) 60:64
    "ekf_q1", "ekf_q2", "ekf_q3", "ekf_q4",
    "ekf_rollspeed_rads", "ekf_pitchspeed_rads", "ekf_yawspeed_rads",
    # ESTIMATOR_STATUS (#230) — innovation test ratios (>1.0 = measurement rejected) 64:68
    "ekf_vel_ratio",        # velocity innovation test ratio
    "ekf_pos_horiz_ratio",  # horizontal position innovation test ratio
    "ekf_pos_vert_ratio",   # vertical position innovation test ratio
    "ekf_mag_ratio",        # magnetometer innovation test ratio
    # (hagl_ratio and tas_ratio omitted — always NaN in multirotor SITL without rangefinder/airspeed)
    # ESTIMATOR_STATUS — 1-σ position accuracy (m) 68:70
    "ekf_pos_horiz_acc_m",
    "ekf_pos_vert_acc_m",
    # ESTIMATOR_STATUS — health / anomaly flags 70:73
    "ekf_flags",        # uint16 ESTIMATOR_STATUS_FLAGS bitfield  (0 = all good)
    "ekf_gps_glitch",   # 1 if GPS glitch detected  (bit 10 = 0x0400)
    "ekf_accel_error",  # 1 if bad accelerometer     (bit 11 = 0x0800)
]
'''
from torch.utils.data import DataLoader, Subset
from base.base_dataset import BaseADDataset
from base.spoofing_dataset_next import MySpoofingPhysical
from base.spoofing_dataset import MySpoofing
from .preprocessing import create_semisupervised_setting
import torch
import os
import logging
import numpy as np
from sklearn.model_selection import train_test_split

from pathlib import Path

class Pegasus(BaseADDataset):
    def __init__(self, root: str, known_outlier_class: tuple = tuple(), n_known_outlier_classes: int = 0, ratio_known_normal: float = 0.0,
                 ratio_known_outlier: float = 0.0, ratio_pollution: float = 0.0, random_state=None):
        super().__init__(root)
        # 0: normal, 1: spoofing, 2: replay, 3: gyro bias,
        # 4: motor fault, 5: motor delay, 6: GPS denial, 7: IMU HF noise
        self.n_classes = 8
        self.normal_classes = (0,)
        self.outlier_classes = (1, 2, 3, 4, 5, 6, 7)
        self.n_anomaly_classes = len(self.outlier_classes)  # 7

        if n_known_outlier_classes == 0:
            self.known_outlier_classes = ()
        else:
            self.known_outlier_classes = known_outlier_class
        self.unknown_outlier_classes = tuple(set(self.outlier_classes) - set(self.known_outlier_classes))

        logger = logging.getLogger()

        # Load data — labels are already 2-D multi-hot: (n_samples, n_anomaly_classes)
        path = os.path.join(root, 'Pegasus')
        data_train = np.load(os.path.join(path, 'dataset.npz'))

        signals      = data_train["features"]
        signals_next = data_train["next"]
        flags_mh     = data_train["labels"].astype(np.float32)  # (n, n_anomaly_classes) multi-hot

        idx_norm = (flags_mh == 0).all(axis=1)
        idx_out = flags_mh.any(axis=1)

        test_ratio = 0.3

        # Split normal samples
        (X_train_norm, X_test_norm,
         fmh_train_norm, fmh_test_norm,
         next_train_norm, next_test_norm) = train_test_split(
            signals[idx_norm], flags_mh[idx_norm], signals_next[idx_norm],
            test_size=test_ratio, random_state=random_state)

        # Split outlier samples
        (X_train_out, X_test_out,
         fmh_train_out, fmh_test_out,
         next_train_out, next_test_out) = train_test_split(
            signals[idx_out], flags_mh[idx_out], signals_next[idx_out],
            test_size=test_ratio, random_state=random_state)

        X_train    = np.concatenate([X_train_norm, X_train_out])
        X_test     = np.concatenate([X_test_norm,  X_test_out])
        y_train    = np.concatenate([fmh_train_norm, fmh_train_out])   # multi-hot
        y_test     = np.concatenate([fmh_test_norm,  fmh_test_out])    # multi-hot
        next_train = np.concatenate([next_train_norm, next_train_out])
        next_test  = np.concatenate([next_test_norm,  next_test_out])

        logger.info(f'n sample in train: Normal: {len(fmh_train_norm)}, '
                    f'anomaly columns sum: {fmh_train_out.sum(axis=0).tolist()}')
        logger.info(f'n sample in test:  Normal: {len(fmh_test_norm)}, '
                    f'anomaly columns sum: {fmh_test_out.sum(axis=0).tolist()}')

        n_train_normal  = int((y_train.sum(axis=1) == 0).sum())
        n_train_anomaly = int((y_train.sum(axis=1) > 0).sum())
        n_test_normal   = int((y_test.sum(axis=1) == 0).sum())
        n_test_anomaly  = int((y_test.sum(axis=1) > 0).sum())
        logger.info(f'n sample in train: Normal: {n_train_normal}, anomaly: {n_train_anomaly}')
        logger.info(f'n sample in test:  Normal: {n_test_normal},  anomaly: {n_test_anomaly}')

        # Construct validation set (50 % of test)
        # Use the passed random_state for the val/test split so that the split
        # is deterministic (not dependent on the global numpy random state).
        val_ratio = 0.5
        _rng = random_state if random_state is not None else np.random
        idx_val = _rng.choice(len(y_test), size=int(val_ratio * len(y_test)), replace=False)
        mask = np.ones(len(y_test), dtype=bool)
        mask[idx_val] = False

        X_val      = X_test[~mask]
        y_val      = y_test[~mask]
        X_test     = X_test[mask]
        y_test     = y_test[mask]
        next_val   = next_test[~mask]
        next_test  = next_test[mask]

        # Training set
        train_set = MySpoofingPhysical(X_train, y_train, next_train)

        # Create semi-supervised setting
        idx, _, semi_targets = create_semisupervised_setting(
            train_set.targets.cpu().numpy(),
            self.known_outlier_classes,
            self.outlier_classes,
            ratio_known_normal, ratio_known_outlier, ratio_pollution
        )
        train_set.semi_targets[idx] = torch.tensor(semi_targets)

        self.X_train  = X_train
        self.y_train  = y_train
        self.semi_y   = semi_targets
        self.X_test   = X_test
        self.y_test   = y_test
        self.X_val    = X_val
        self.y_val    = y_val

        self.train_set = Subset(train_set, idx)
        self.val_set   = MySpoofingPhysical(X_val, y_val, next_val)
        self.test_set  = MySpoofingPhysical(X_test, y_test, next_test)

    def loaders(self, batch_size: int, shuffle_train=True, shuffle_test=False, num_workers: int = 0) -> tuple[DataLoader, DataLoader]:
        train_loader = DataLoader(dataset=self.train_set, batch_size=batch_size, shuffle=shuffle_train,
                                  num_workers=num_workers, drop_last=True)
        val_loader   = DataLoader(dataset=self.val_set,   batch_size=batch_size, shuffle=shuffle_test,
                                  num_workers=num_workers, drop_last=False)
        test_loader  = DataLoader(dataset=self.test_set,  batch_size=batch_size, shuffle=shuffle_test,
                                  num_workers=num_workers, drop_last=False)
        return train_loader, val_loader, test_loader

    def data_direct(self):
        return self.X_train, self.y_train, self.semi_y, self.X_test, self.y_test, self.X_val, self.y_val
