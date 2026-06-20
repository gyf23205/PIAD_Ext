import os
import numpy as np
import pandas as pd
from pathlib import Path
from glob import glob

data_path = "data/ALFA/all_flights/"
time_column = "%time"
timestamp_column = "timestamp"

def read_data(full_path):
    df_tmp = pd.read_csv(full_path)
    df_tmp = df_tmp.rename(columns={"%time": timestamp_column})

    df_tmp[timestamp_column] = pd.to_datetime(df_tmp[timestamp_column], unit="ns")
    df_tmp.set_index(timestamp_column, inplace=True)
    return df_tmp

def extract_topic_name(flight_name, file_name):
    topic_name = file_name.split(flight_name)
    topic_name = topic_name[1]
    topic_name = topic_name[1:]
    topic_name = topic_name.split(".csv")
    topic_name = topic_name[0] 
    return topic_name

if __name__ == "__main__":
    flight_topic_dict = {}
    topic_list = []
    df_dict = {}
    window_size = 40  # 5 seconds window at 5Hz frequency
    slide_size = 2   # Slide by 1 time step

    # unused_flight_list = ["aileron", "aileron_failure", "elevator", 
    #                     "no_ground_truth", "rudder"]
    
    unused_flight_list = []

    # NOTE: mavros-rc-out (servo PWM outputs) is kept — it is the most direct
    # per-control-surface actuator signal and is essential for telling aileron /
    # elevator / rudder failures apart (multi-hot fault-type classification).
    # mavros-rc-in, mavctrl-rpy and setpoint_raw-local are still excluded:
    # rc-in is mostly constant pilot channels, and mavctrl-rpy / setpoint_raw-local
    # are present in only 40/46 flights so including them would misalign features.
    unused_topic_list = ["diagnostics", "emergency_responder-traj_file", "global_position",
                        "local_position", "mavctrl-path_dev",
                        "mavctrl-rpy", "mavlink",
                        "mavros-battery", "mavros-imu-data_raw",
                        "mavros-imu-mag", "mavros-mission-reached",
                        "mavros-rc-in", "mavros-state",
                        "mavros-time_reference", "setpoint_raw"] # Using failure_status topic

    # NOTE: field.commanded is kept (was previously dropped). For nav_info-roll /
    # -pitch / -yaw / -airspeed the commanded-vs-measured residual is the defining
    # signature of a control-surface fault, so it must reach the model.
    unused_columns = ["field.header.seq", "field.header.stamp", "field.header.frame_id",
                    "field.variance", "field.twist.angular.x",
                    "field.twist.angular.y", "field.twist.angular.z",
                    "field.coordinate_frame"]

    # unused_columns = ["field.header.seq", "field.header.stamp", "field.header.frame_id",
    #                   "field.commanded", "field.variance", "field.twist.angular.x",
    #                   "field.twist.angular.y", "field.twist.angular.z",
    #                   "field.coordinate_frame", "mavros-nav_info-airspeed.measured",
    #                   "mavros-vfr_hud.airspeed", "mavros-vfr_hud.throttle",
    #                   "mavros-vfr_hud.altitude", "mavros-imu-atm_pressure.fluid_pressure",
    #                   "mavros-imu-data.angular_velocity.z", "mavros-imu-data.linear_acceleration.z",
    #                   "mavros-imu-data.orientation.w", "mavros-imu-data.orientation.z",
    #                   "mavros-imu-temperature.temperature", "mavros-nav_info-velocity.des_x",
    #                   "mavros-nav_info-velocity.des_y", "mavros-nav_info-velocity.des_z",
    #                   "mavros-nav_info-velocity.meas_x", "mavros-nav_info-velocity.meas_z",
    #                   "mavros-wind_estimation.twist.linear.y", "mavros-wind_estimation.twist.linear.z",
    #                   "mavros-vfr_hud.groundspeed"]

    n_normal1 = 0
    n_anomaly1 = 0

    n_normal_file = 0
    n_anomaly_file = 0

    n_rows_normal = []
    n_rows_anomaly = []

    # Iterate over the list of flight names
    for i, flight in enumerate(glob(os.path.join(data_path + "*"))):
        if any(x in flight for x in unused_flight_list):
            continue
        # print(i, flight)
        flight_name = os.path.basename(flight)

        if flight_name not in flight_topic_dict:
            flight_topic_dict[flight_name] = []
        
        df_merged = None
        
        # Iterate over the list of topics
        for k, topic in enumerate(glob(flight + "/*.csv")):
            if any(x in topic for x in unused_topic_list):
                continue
            
            file_name = os.path.basename(topic)
            topic_list.append(file_name)
            topic_name = extract_topic_name(flight_name, file_name)
            flight_topic_dict[flight_name].append(topic_name)
            
            # Drop columns
            dfx = read_data(topic)
            dfx = dfx.drop(unused_columns, axis=1, errors="ignore")
            new_columns = list(map(lambda x: f"{topic_name}.{x.replace('field.', '')}", dfx.columns))
            dfx.columns = new_columns  # Direct assignment instead of set_axis
            # Drop all covariance columns
            dfx = dfx.drop(dfx.filter(regex='covariance').columns, axis=1)
            
            # Resample the dataset to 5Hz frequency 
            dfx = dfx.resample("200ms").median()

            # Special handling for the failure_status
            if "failure_status" in topic_name:
                # Give different labels for different failure types
                if "engine" in topic_name: # Keep the engine failure status as 1
                    pass
                elif "aileron" in topic_name: # Aileron failure will be labeled 2, 3. 3 is missing from single failure dataset and are contained in hybrid failure dataset
                    if flight_name == "carbonZ_2018-09-11-17-27-13_2_both_ailerons_failure":
                        dfx[topic_name + ".data"] = dfx[topic_name + ".data"].replace([3], [8]) # 7 means both ailerons fail
                    elif flight_name == "carbonZ_2018-09-11-17-27-13_1_rudder_zero__left_aileron_failure":
                        dfx[topic_name + ".data"] = dfx[topic_name + ".data"].replace([2], [9])
                    else:
                        dfx[topic_name + ".data"] = dfx[topic_name + ".data"].replace([1, 2], [2, 3])

                elif "elevator" in topic_name: # Elevator failure will be labeled 4
                    dfx[topic_name + ".data"] = dfx[topic_name + ".data"].replace([1], [4])

                elif "rudder" in topic_name: # Rudder failure will be labeled 5, 6. 1 is missing from single failure dataset and are contained in hybrid failure dataset
                    dfx[topic_name + ".data"] = dfx[topic_name + ".data"].replace([2, 3], [5, 6])

            if df_merged is None:
                df_merged = dfx
            else:
                df_merged = df_merged.merge(dfx, left_index=True, right_index=True, how="outer")
            df_merged = df_merged.ffill().bfill()


        df_merged = df_merged.drop(unused_columns, axis=1, errors="ignore")
        df_dict[flight_name] = df_merged
        if "no_failure" in flight_name:
            n_normal_file += 1
            n_normal1 += df_merged.values.shape[0]
            n_rows_normal.append(df_merged.values.shape[0])
        else:
            n_anomaly_file += 1
            n_anomaly1 += df_merged.values.shape[0]
            n_rows_anomaly.append(df_merged.values.shape[0])

    print("normal: ", n_normal1)
    print("anoamly: ", n_anomaly1)

    print(n_normal_file)
    print(n_anomaly_file)

    print(n_rows_normal)
    print(n_rows_anomaly)

    print(np.mean(n_rows_normal))
    print(np.mean(n_rows_anomaly))

    X_all = []
    y_all = []
    next_all = []

    # Convert to numpy arrays
    for flight_name, df in df_dict.items():      
        # Extract features
        X = df.drop(columns=[col for col in df.columns if "status" in col]).values

        n_windows = (len(X) - window_size) // slide_size
        starts = np.arange(n_windows) * slide_size
        ends = starts + window_size
        X_win = np.stack([X[starts[i]:ends[i]] for i in range(n_windows)])
        X_next = X[ends, :]

        # Extract labels.
        if "no_failure" in flight_name:
            y_win = np.zeros(n_windows)
        else:
            for col in df.columns:
                if "status" in col:
                    y = df[col].values
                    break
            y_win = np.array([max(y[starts[i]:ends[i]]) for i in range(n_windows)])

        X_all.append(X_win)
        y_all.append(y_win)
        next_all.append(X_next)

    X_all = np.concatenate(X_all, axis=0)
    y_all = np.concatenate(y_all, axis=0)
    next_all = np.concatenate(next_all, axis=0)

    # Normalize the features
    mu = np.mean(X_all, axis=(0,1))
    std = np.std(X_all, axis=(0,1))
    eps = 1e-7
    X_all = (X_all - mu) / (std + eps)
    next_all = (next_all - mu) / (std + eps)

    print(f"Final dataset shape: {X_all.shape}, Labels shape: {y_all.shape}, Next shape: {next_all.shape}")
    print(f"Num of normal samples: {np.sum(y_all==0)}, Num of anomaly samples: {np.sum(y_all!=0)}")

    # Save the final numpy arrays
    np.save("data/ALFA/X_all.npy", X_all)
    np.save("data/ALFA/y_all.npy", y_all)
    np.save("data/ALFA/next_all.npy", next_all)
