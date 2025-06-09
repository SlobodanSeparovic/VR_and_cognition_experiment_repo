import numpy as np
import pandas as pd

def calculate_kinematics(stream, start_time, end_time, buffer):
    """Calculate kinematics"""
    # Stream to df
    motion_cols = [
        'HmdPosX', 'HmdPosY', 'HmdPosZ', 'HmdRotX', 'HmdRotY', 'HmdRotZ', 'HmdRotW',
        'LeftPosX', 'LeftPosY', 'LeftPosZ', 'LeftRotX', 'LeftRotY', 'LeftRotZ', 'LeftRotW',
        'RightPosX', 'RightPosY', 'RightPosZ', 'RightRotX', 'RightRotY', 'RightRotZ', 'RightRotW',
        'LeftTrigger', 'RightTrigger'
    ]
    df = pd.DataFrame(stream["time_series"], columns=motion_cols)
    df['Timestamp'] = stream["time_stamps"]
    
    # Crop to task duration
    df_cropped = df[(df['Timestamp'] >= start_time) & (df['Timestamp'] <= end_time)].copy()
    
    # Convert motion timestamps to relative time (like EEG for alignment)
    df_cropped['Timestamp'] = df_cropped['Timestamp'] - df_cropped['Timestamp'].min()

    # Calculate velocities (first row will have NaN from buffer pre-task)
    devices = ['Hmd', 'Left', 'Right']
    for device in devices:
        df_cropped = _calc_linear_velocity(df_cropped, device)
        df_cropped = _calc_angular_velocity(df_cropped, device)
        
    # Calculate thresholds and flag artifacts
    thresholds = _calculate_thresholds(df_cropped)
    df_cropped = _flag_artifacts(df_cropped, thresholds, buffer)

    # Remove raw position/rotation columns
    df_cropped = df_cropped.drop(columns=[c for c in df_cropped.columns if 'Pos' in c or 'Rot' in c])

    # Remove initial NaN (this reflects less than 0.01 s of motion data loss)
    df_cropped = df_cropped.iloc[1:].reset_index(drop=True)

    return df_cropped

def _calc_linear_velocity(df, device):
    """Calculate 3D linear velocity in m/s from positional data"""
    pos_cols = [f'{device}PosX', f'{device}PosY', f'{device}PosZ']
    
    # Calculate actual time differences between samples
    time_deltas = df['Timestamp'].diff()
    
    # Compute average delta using actual differences (not timestamp mean)
    avg_delta = time_deltas.mean()
    
    # Handle missing/zero values using real average delta
    time_deltas = time_deltas.fillna(avg_delta).replace(0, avg_delta)
    
    # Calculate positional displacement (delta x, delta y, delta z)
    displacement = df[pos_cols].diff()
    
    # Compute velocity magnitude (norm of displacement / time)
    df[f'{device}Velocity'] = np.linalg.norm(displacement, axis=1) / time_deltas
    return df

def _calc_angular_velocity(df, device):
    """Calculate angular velocity from quaternion differences (radians/second)"""
    quat_cols = [f'{device}RotX', f'{device}RotY', f'{device}RotZ', f'{device}RotW']
    
    # Normalize quaternions to ensure unit length
    quaternions = df[quat_cols].to_numpy()
    norms = np.linalg.norm(quaternions, axis=1, keepdims=True)
    quaternions = np.divide(quaternions, norms, where=(norms != 0))  # Avoid division by zero
    
    # Calculate angular displacement
    q1 = quaternions[:-1]
    q2 = quaternions[1:]
    dot_products = np.sum(q1 * q2, axis=1)
    
    # Safe arccos with clipping to avoid numerical errors
    angles = 2 * np.arccos(np.clip(np.abs(dot_products), -1.0, 1.0))  # Radians
    
    # Calculate time deltas (proper average calculation)
    time_deltas = df['Timestamp'].diff()
    avg_delta = time_deltas.mean()  # Actual average time difference
    time_deltas = time_deltas.fillna(avg_delta).replace(0, avg_delta)
    
    # Calculate angular velocity
    angular_velocity = np.full_like(time_deltas, np.nan)
    angular_velocity[1:] = angles / time_deltas[1:]  # Explicit index alignment
    
    df[f'{device}AngularVelocity'] = angular_velocity
    return df

def _calculate_thresholds(df, percentile=99.9):
    """Calculate motion change thresholds using specified percentile value"""
    thresholds = {}
    devices = ['Hmd', 'Left', 'Right']
    for device in devices:
        lin_vel = df[f'{device}Velocity'].dropna()
        ang_vel = df[f'{device}AngularVelocity'].dropna()
        thresholds[f'{device}_lin'] = np.percentile(lin_vel, percentile)
        thresholds[f'{device}_ang'] = np.percentile(ang_vel, percentile)
    return thresholds

def _flag_artifacts(df, thresholds, buffer):
    """Flag artifacts with buffers, incuding inital buffer handled separately as task was initiated with a trigger"""
    df = df.copy()
    df['MotionArtifact'] = False
    
    # Initial asymmetric buffer (handled separately as no buffer needs to be added to it later)
    start_time = df['Timestamp'].min()
    initial_end = start_time + (buffer * 0.85)
    initial_mask = (df['Timestamp'] >= start_time) & (df['Timestamp'] <= initial_end)

    # Motion/trigger artifacts (exclude initial buffer from this logic)
    other_artifacts = pd.Series(False, index=df.index)
    for device in ['Hmd', 'Left', 'Right']:
        lin_mask = df[f'{device}Velocity'] > thresholds[f'{device}_lin']
        ang_mask = df[f'{device}AngularVelocity'] > thresholds[f'{device}_ang']
        other_artifacts |= lin_mask | ang_mask
    
    for side in ['Left', 'Right']:
        trigger_col = df[f'{side}Trigger']
        # Detect trigger onsets (button press)
        trigger_onset = (trigger_col.diff() > 0.5)
        other_artifacts |= trigger_onset.fillna(False)
        # Detect trigger offsets (button release)
        trigger_offset = (trigger_col.diff() < -0.5)
        other_artifacts |= trigger_offset.fillna(False)
    
    # Apply asymmetric buffer only to non-initial artifacts
    artifact_times = df[other_artifacts]['Timestamp']
    buffered_other_artifacts = pd.Series(False, index=df.index)
    for t in artifact_times:
        buffered_other_artifacts |= (df['Timestamp'] >= t - (buffer * 0.2)) & (df['Timestamp'] <= t + (buffer * 0.85))
    
    # Combine masks (initial buffer + other buffered artifacts)
    df['MotionArtifact'] = initial_mask | buffered_other_artifacts

    return df