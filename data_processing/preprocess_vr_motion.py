import numpy as np
import pandas as pd

def calculate_kinematics(stream, start_time, end_time, buffer):
    """
    Calculate kinematics with a buffer period.
    
    Parameters:
        buffer (float): Seconds before task start to include for velocity calculations.
    """
    # Stream to df
    motion_cols = [
        'HmdPosX', 'HmdPosY', 'HmdPosZ', 'HmdRotX', 'HmdRotY', 'HmdRotZ', 'HmdRotW',
        'LeftPosX', 'LeftPosY', 'LeftPosZ', 'LeftRotX', 'LeftRotY', 'LeftRotZ', 'LeftRotW',
        'RightPosX', 'RightPosY', 'RightPosZ', 'RightRotX', 'RightRotY', 'RightRotZ', 'RightRotW',
        'LeftTrigger', 'RightTrigger'
    ]
    df = pd.DataFrame(stream["time_series"], columns=motion_cols)
    df['Timestamp'] = stream["time_stamps"]
    
    # Crop with buffer before task start
    df_cropped = df[
        (df['Timestamp'] >= start_time - buffer) & 
        (df['Timestamp'] <= end_time)
    ].copy()

    # Calculate velocities (first row will have NaN from buffer pre-task)
    devices = ['Hmd', 'Left', 'Right']
    for device in devices:
        df_cropped = _calc_linear_velocity(df_cropped, device)
        df_cropped = _calc_angular_velocity(df_cropped, device)

    # Remove raw position/rotation columns
    df_cropped = df_cropped.drop(columns=[c for c in df_cropped.columns if 'Pos' in c or 'Rot' in c])

    # Remove initial NaN (even if buffer = 0, this reflects less than 0.01 s of motion data loss)
    df_cropped = df_cropped.iloc[1:].reset_index(drop=True)

    return df_cropped

def _calc_linear_velocity(df, device):
    """Calculate 3D linear velocity in m/s from positional data."""
    pos_cols = [f'{device}PosX', f'{device}PosY', f'{device}PosZ']
    
    # Calculate actual time differences between samples
    time_deltas = df['Timestamp'].diff()
    
    # Compute average delta using actual differences (not timestamp mean)
    avg_delta = time_deltas.mean()
    
    # Handle missing/zero values using real average delta
    time_deltas = time_deltas.fillna(avg_delta).replace(0, avg_delta)
    
    # Calculate positional displacement (Δx, Δy, Δz)
    displacement = df[pos_cols].diff()
    
    # Compute velocity magnitude (norm of displacement / time)
    df[f'{device}Velocity'] = np.linalg.norm(displacement, axis=1) / time_deltas
    return df

def _calc_angular_velocity(df, device):
    """Calculate angular velocity from quaternion differences (radians/second)."""
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
    angular_velocity = np.concatenate([[np.nan], angles]) / time_deltas
    
    df[f'{device}AngularVelocity'] = angular_velocity
    return df