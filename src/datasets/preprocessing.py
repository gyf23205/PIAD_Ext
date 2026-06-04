import numpy as np
import os
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import logging
import pickle

def create_semisupervised_setting(labels, *args):
    """
    Dispatch to scalar (legacy) or multi-hot (new) semi-supervised setting creation.

    **Multi-hot form** (new, used by ALFA / Pegasus):
        create_semisupervised_setting(labels_2d, known_outlier_classes, outlier_classes,
                                      ratio_known_normal, ratio_known_outlier, ratio_pollution)
        labels_2d : (n_samples, n_anomaly_classes) float array — multi-hot, columns ordered by outlier_classes.

    **Scalar form** (legacy, kept for backward compatibility):
        create_semisupervised_setting(labels_1d, normal_classes, unknown_outlier_classes,
                                      known_outlier_classes, ratio_known_normal, ratio_known_outlier,
                                      ratio_pollution)
        labels_1d : (n_samples,) int array — one class index per sample.
    """
    labels = np.array(labels)
    if labels.ndim == 2:
        # New multi-hot path
        known_outlier_classes, outlier_classes, r1, r2, r3 = args
        return _create_semisupervised_setting_multihot(labels, known_outlier_classes, outlier_classes, r1, r2, r3)
    else:
        # Legacy scalar path
        normal_classes, unknown_outlier_classes, known_outlier_classes, r1, r2, r3 = args
        return _create_semisupervised_setting_scalar(labels, normal_classes, unknown_outlier_classes,
                                                     known_outlier_classes, r1, r2, r3)


def _create_semisupervised_setting_scalar(labels, normal_classes, unknown_outlier_classes, known_outlier_classes,
                                          ratio_known_normal, ratio_known_outlier, ratio_pollution):
    """
    Legacy scalar-label semi-supervised setting (original implementation).
    Returns: (list_idx, list_labels, list_semi_labels)  — all 1-D / lists of scalars.
    Semi-label convention: -1 unlabeled, 0 labeled normal, k>0 labeled anomaly class k.
    """
    logger = logging.getLogger()

    idx_normal = np.argwhere(np.isin(labels, normal_classes)).flatten()
    idx_unknown_outlier = np.argwhere(np.isin(labels, unknown_outlier_classes)).flatten()
    idx_known_outlier = np.argwhere(np.isin(labels, known_outlier_classes)).flatten()
    idx_all_outlier = np.concatenate((idx_unknown_outlier, idx_known_outlier), axis=0)

    n_normal = len(idx_normal)

    a = np.array([[1, 1, 0, 0],
                  [(1-ratio_known_normal), -ratio_known_normal, -ratio_known_normal, -ratio_known_normal],
                  [-ratio_known_outlier, -ratio_known_outlier, -ratio_known_outlier, (1-ratio_known_outlier)],
                  [0, -ratio_pollution, (1-ratio_pollution), 0]])
    b = np.array([n_normal, 0, 0, 0])
    x = np.linalg.solve(a, b)

    n_labeled_normal   = int(x[0])
    n_unlabeled_normal = int(x[1])
    n_unlabeled_outlier = int(x[2])
    n_labeled_outlier  = int(x[3])
    logger.info(f'In semi setting: n_labeled_normal: {n_labeled_normal}, n_unlabeled_normal: {n_unlabeled_normal}, n_unlabeled_outlier: {n_unlabeled_outlier}, n_labeled_outlier: {n_labeled_outlier}')

    perm_normal          = np.random.permutation(n_normal)
    perm_labeled_outlier = np.random.permutation(len(idx_known_outlier))

    idx_labeled_normal   = idx_normal[perm_normal[:n_labeled_normal]].tolist()
    idx_unlabeled_normal = idx_normal[perm_normal[n_labeled_normal:n_labeled_normal+n_unlabeled_normal]].tolist()
    idx_labeled_outlier  = idx_known_outlier[perm_labeled_outlier[:n_labeled_outlier]].tolist()
    idx_unlabeled_outlier = np.setdiff1d(idx_all_outlier, idx_labeled_outlier)
    perm_unlabeled_outlier = np.random.permutation(len(idx_unlabeled_outlier))
    idx_unlabeled_outlier = idx_unlabeled_outlier[perm_unlabeled_outlier[:n_unlabeled_outlier]].tolist()

    labels_labeled_normal   = labels[idx_labeled_normal].tolist()
    labels_unlabeled_normal = labels[idx_unlabeled_normal].tolist()
    labels_unlabeled_outlier = labels[idx_unlabeled_outlier].tolist()
    labels_labeled_outlier  = labels[idx_labeled_outlier].tolist()

    known_outlier_classes_in_labeled = np.unique(labels_labeled_outlier)
    missing_classes = np.setdiff1d(known_outlier_classes, known_outlier_classes_in_labeled)
    for mc in missing_classes:
        idx_sample_mc = np.argwhere(labels == mc).flatten()[0]
        labels_labeled_outlier.append(mc)
        idx_labeled_outlier.append(idx_sample_mc)

    semi_labels_labeled_normal   = np.zeros(n_labeled_normal).astype(np.int32).tolist()
    semi_labels_unlabeled_normal = (-np.ones(n_unlabeled_normal)).astype(np.int32).tolist()
    semi_labels_unlabeled_outlier = (-np.ones(n_unlabeled_outlier)).astype(np.int32).tolist()
    semi_labels_labeled_outlier  = labels_labeled_outlier

    list_idx = idx_labeled_normal + idx_unlabeled_normal + idx_unlabeled_outlier + idx_labeled_outlier
    list_labels = labels_labeled_normal + labels_unlabeled_normal + labels_unlabeled_outlier + labels_labeled_outlier
    list_semi_labels = (semi_labels_labeled_normal + semi_labels_unlabeled_normal
                        + semi_labels_unlabeled_outlier + semi_labels_labeled_outlier)

    return list_idx, list_labels, list_semi_labels


def _create_semisupervised_setting_multihot(labels, known_outlier_classes, outlier_classes,
                                            ratio_known_normal, ratio_known_outlier, ratio_pollution):
    """
    Multi-hot semi-supervised setting for multi-label anomaly detection.
    :param labels: 2D np.array (n_samples, n_anomaly_classes) — multi-hot ground-truth.
    :param known_outlier_classes: anomaly class numbers labeled during training.
    :param outlier_classes: ordered list of ALL anomaly class numbers (defines column order).
    :return: (list_idx, list_labels_2d, semi_labels_2d)
        semi_labels_2d:  all -1 → unlabeled | all 0 → labeled normal | some 1s → labeled anomaly
    """
    logger = logging.getLogger()

    labels = np.array(labels)
    n_anomaly_classes = labels.shape[1]
    outlier_classes = list(outlier_classes)
    known_outlier_classes = list(known_outlier_classes)

    # Column indices in the multi-hot array for each known anomaly class
    known_cols = [outlier_classes.index(k) for k in known_outlier_classes]

    # Identify sample groups
    is_normal = (labels.sum(axis=1) == 0)
    # Known outlier: has at least one known-class bit set
    has_known = labels[:, known_cols].any(axis=1) if len(known_cols) > 0 else np.zeros(len(labels), dtype=bool)
    is_known_outlier = (~is_normal) & has_known
    is_unknown_outlier = (~is_normal) & (~has_known)

    idx_normal          = np.where(is_normal)[0]
    idx_known_outlier   = np.where(is_known_outlier)[0]
    idx_unknown_outlier = np.where(is_unknown_outlier)[0]
    idx_all_outlier     = np.concatenate((idx_unknown_outlier, idx_known_outlier))

    n_normal = len(idx_normal)

    # Solve system of linear equations to obtain respective number of samples
    a = np.array([[1, 1, 0, 0],
                  [(1-ratio_known_normal), -ratio_known_normal, -ratio_known_normal, -ratio_known_normal],
                  [-ratio_known_outlier, -ratio_known_outlier, -ratio_known_outlier, (1-ratio_known_outlier)],
                  [0, -ratio_pollution, (1-ratio_pollution), 0]])
    b = np.array([n_normal, 0, 0, 0])
    x = np.linalg.solve(a, b)

    # Get number of samples
    n_labeled_normal   = int(x[0])
    n_unlabeled_normal = int(x[1])
    n_unlabeled_outlier = int(x[2])
    n_labeled_outlier  = int(x[3])
    logger.info(f'In semi setting: n_labeled_normal: {n_labeled_normal}, n_unlabeled_normal: {n_unlabeled_normal}, '
                f'n_unlabeled_outlier: {n_unlabeled_outlier}, n_labeled_outlier: {n_labeled_outlier}')

    # Sample indices
    perm_normal          = np.random.permutation(n_normal)
    perm_labeled_outlier = np.random.permutation(len(idx_known_outlier))

    idx_labeled_normal   = idx_normal[perm_normal[:n_labeled_normal]].tolist()
    idx_unlabeled_normal = idx_normal[perm_normal[n_labeled_normal:n_labeled_normal+n_unlabeled_normal]].tolist()
    idx_labeled_outlier  = idx_known_outlier[perm_labeled_outlier[:n_labeled_outlier]].tolist()
    # Exclude already-labeled outliers from the unlabeled pool
    idx_unlabeled_outlier = np.setdiff1d(idx_all_outlier, idx_labeled_outlier)
    perm_unlabeled_outlier = np.random.permutation(len(idx_unlabeled_outlier))
    idx_unlabeled_outlier = idx_unlabeled_outlier[perm_unlabeled_outlier[:n_unlabeled_outlier]].tolist()

    # Guard: ensure every known outlier class appears at least once in the labeled set
    for cls in known_outlier_classes:
        col = outlier_classes.index(cls)
        labeled_out_arr = np.array(idx_labeled_outlier)
        if len(labeled_out_arr) == 0 or not labels[labeled_out_arr, col].any():
            candidates = np.where(labels[:, col] == 1)[0]
            if len(candidates) > 0:
                idx_labeled_outlier.append(int(candidates[0]))

    # ---- Build semi-supervised labels (2D multi-hot) ----
    # Labeled normal: all-zero rows
    semi_labeled_normal   = np.zeros((len(idx_labeled_normal), n_anomaly_classes), dtype=np.float32)
    # Unlabeled (normal and unknown outlier): all-(-1) rows
    semi_unlabeled_normal  = -np.ones((len(idx_unlabeled_normal), n_anomaly_classes), dtype=np.float32)
    semi_unlabeled_outlier = -np.ones((len(idx_unlabeled_outlier), n_anomaly_classes), dtype=np.float32)
    # Labeled anomaly: copy only the known-class columns from the actual labels
    semi_labeled_outlier = np.zeros((len(idx_labeled_outlier), n_anomaly_classes), dtype=np.float32)
    if len(idx_labeled_outlier) > 0 and len(known_cols) > 0:
        semi_labeled_outlier[:, known_cols] = labels[idx_labeled_outlier][:, known_cols]

    # ---- Assemble final lists ----
    list_idx = idx_labeled_normal + idx_unlabeled_normal + idx_unlabeled_outlier + idx_labeled_outlier

    list_labels = np.concatenate([
        labels[idx_labeled_normal],
        labels[idx_unlabeled_normal],
        labels[idx_unlabeled_outlier],
        labels[idx_labeled_outlier],
    ], axis=0)  # (total, n_anomaly_classes)

    semi_labels_array = np.concatenate([
        semi_labeled_normal,
        semi_unlabeled_normal,
        semi_unlabeled_outlier,
        semi_labeled_outlier,
    ], axis=0)  # (total, n_anomaly_classes)

    return list_idx, list_labels, semi_labels_array

def normalization(data):
    mu = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    eps = 1e-7
    data_std = (data - mu) / (std + eps)
    data_scaled = (data_std - np.min(data_std, axis=0))/(np.max(data_std, axis=0) - np.min(data_std, axis=0) + eps)
    params = {'mu':mu, 'std':std, 'min':np.min(data_std, axis=0), 'max':np.max(data_std, axis=0)}
    with open('/home/yifan/git/qualisys_drone_sdk/examples/params.pkl', 'wb') as f:
        pickle.dump(params, f)
    return data_scaled

def batch_sequential():
    seq_len = 100
    signal_path = os.path.join(source,'traj_log.npy')
    flags_path = os.path.join(source,'flag_log.npy')
    signals = np.load(signal_path)[:, 0:12]
    flags = np.load(flags_path)
    print(signals.shape)
    n_channels = signals.shape[-1]

    signals_scaled = normalization(signals)
    signals_batched = np.zeros((len(signals_scaled)-seq_len+1, seq_len, n_channels))
    flags_batched = np.zeros((len(signals_scaled)-seq_len+1, seq_len))

    for i in range(len(signals_scaled)-seq_len+1):
        signals_batched[i,:,:] = signals_scaled[i:i+seq_len,:]
        flags_batched[i, :] = flags[i:i+seq_len]
    np.save(os.path.join(root, 'data_wind_seq.npy'), signals_batched)
    np.save(os.path.join(root, 'labels_wind_seq.npy'),flags_batched)

def build_next(root):
    signal_path = os.path.join(root,'data_unscaled_multi_noise.npy')
    flags_path = os.path.join(root,'labels_unscaled_multi_noise.npy')
    signals = np.squeeze(np.load(signal_path))
    flags = np.load(flags_path)
    seq_len = 100
    n_samples = signals.shape[0] - 1
    n_channels = 12
    next_real = np.empty((n_samples, n_channels))
    pad = np.zeros((seq_len - 1,))
    flags = np.concatenate([pad, flags], axis=0)
    labels = np.empty((n_samples,))
    for i in range(n_samples):
        next_real[i, :] = signals[i+1, -12:]
        labels[i] = 1 if np.sum(flags[i:i+seq_len]) > 0 else 0
    np.save(os.path.join(root, 'data_real.npy'), signals[:-1, :])
    np.save(os.path.join(root, 'next_real.npy'), next_real)
    np.save(os.path.join(root, 'labels_real.npy'),labels)


def batch_sequential_flat():
    seq_len = 100
    signal_path = os.path.join(source,'traj_log.npy')
    flags_path = os.path.join(source,'flag_log.npy')
    signals = np.load(signal_path)[:, 0:12]
    flags = np.load(flags_path)
    num_channels = signals.shape[-1]
    print(signals.shape)

    signals_scaled = normalization(signals)
    # Each sample move forward one time step
    num_samples = len(signals_scaled)-seq_len
    num_state = 3
    signals_batched = np.zeros((num_samples, seq_len*num_channels))
    signals_next_batched = np.zeros((num_samples, num_channels))
    labels_batched = np.zeros(num_samples)
    for i in range(num_samples):
        signals_batched[i,:] = signals_scaled[i:i+seq_len,:].reshape((seq_len*num_channels,))
        signals_next_batched[i, :] = signals_scaled[i+seq_len,:]
        labels_batched[i] = 1 if np.sum(flags[i:i+seq_len])>0 else 0
    np.save(os.path.join(target, 'data_wind.npy'), signals_batched)
    np.save(os.path.join(target, 'next_wind.npy'), signals_next_batched)

def batch_sequential_flat_state_only():
    seq_len = 100
    signal_path = os.path.join(source,'traj_log.npy')
    flags_path = os.path.join(source,'flag_log.npy')
    signals = np.load(signal_path)[:, 0:12]
    flags = np.load(flags_path)
    num_channels = signals.shape[-1]
    print(signals.shape)
   
    signals_scaled = normalization(signals)
    # Each sample move forward one time step
    num_samples = len(signals_scaled)-seq_len
    num_state = 3
    signals_batched = np.zeros((num_samples, seq_len*num_channels))
    signals_next_batched = np.zeros((num_samples, num_state))
    labels_batched = np.zeros(num_samples)
    for i in range(num_samples):
        signals_batched[i,:] = signals_scaled[i:i+seq_len,:].reshape((seq_len*num_channels,))
        signals_next_batched[i, :] = signals_scaled[i+seq_len, 3:6]
        labels_batched[i] = 1 if np.sum(flags[i:i+seq_len])>0 else 0
    np.save(os.path.join(target, 'data_wind_state_only.npy'), signals_batched)
    np.save(os.path.join(target, 'next_wind_state_only.npy'), signals_next_batched)
    np.save(os.path.join(target, 'labels_wind_state_only.npy'),labels_batched)


if __name__=='__main__':
    source = '' # 
    target = '/home/yifan/git/FIAD/data/spoofing'
    root = '/home/yifan/git/FIAD/data/spoofing'

    batch_sequential_flat_state_only()
    batch_sequential_flat()