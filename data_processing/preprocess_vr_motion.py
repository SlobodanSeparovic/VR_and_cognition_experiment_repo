import numpy as np
import pandas as pd

def calculate_kinematics(df):
    devices = ['Hmd', 'Left', 'Right']
    for device in devices:
        df = _calc_linear_velocity(df, device)
        df = _calc_angular_velocity(df, device)
    # Drop position and rotation columns
    df = df.drop(columns=[c for c in df.columns if 'Pos' in c or 'Rot' in c])
    # Drop the first row (index 0) due to NaNs from differencing
    df = df.iloc[1:].reset_index(drop=True)
    return df

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