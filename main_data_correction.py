import os
import pandas as pd
import numpy as np
from pathlib import Path

def process_all_participants(results_dir='results'):
    """Process all participant data and compile final dataset"""
    final_data = []
    motion_artifacts_data = []

    # Process each participant directory
    for participant_dir in Path(results_dir).iterdir():
        if not participant_dir.is_dir():
            continue
            
        participant_id = participant_dir.name
        input_file = participant_dir / f"{participant_id}_data.csv"
        output_file = participant_dir / f"{participant_id}_data_corrected.csv"

        if not input_file.exists():
            continue

        # Process individual participant file
        corrected_df, motion_artifacts_df = _process_participant_file(input_file)
        if corrected_df.empty:
            continue

        # Save corrected data for individual participant
        corrected_df.to_csv(output_file, index=False)
        print(f"Saved corrected data for {participant_id} to {output_file}")

        # Collect data for final datasets
        motion_artifacts_data.append(motion_artifacts_df)
        final_data.append(corrected_df)

    # Compile final datasets
    if final_data:
        final_df = pd.concat(final_data, ignore_index=True)
        final_path = Path(results_dir) / 'final_dataset.csv'
        
        # Define and enforce column order
        column_order = [
            'Subject', 'Env_Control', 'Env_Nature', 'Env_Urban', 'N', 'NasaCLI', 'FTCLI_corrected',
            'FTPACLI_corrected', 'EngI_corrected', 'F1', 'TotalPress',
            'IncorrNo', 'CorrRatio', 'RTCorrAv', 'RTIncorrAv'
        ]
        final_df = final_df[[col for col in column_order if col in final_df.columns]]
        
        # Update final dataset with conflict resolution
        _update_final_dataset(final_path, final_df)

        # Handle motion artifacts data
        motion_artifacts_df = pd.concat(motion_artifacts_data, ignore_index=True)
        motion_path = Path(results_dir) / 'motion_artifacts.csv'
        _update_final_dataset(motion_path, motion_artifacts_df)
        
        print(f"Final dataset updated at {final_path}")
        print(f"Motion artifacts dataset updated at {motion_path}")
    else:
        print("No participant data found for processing")

def _process_participant_file(file_path):
    """Process individual participant CSV file"""
    df = pd.read_csv(file_path)
    
    # Clean subject ID
    df['Subject'] = df['Subject'].str.replace('P', '').astype(int)
    
    # One-hot encode the 'Env' column
    df = pd.get_dummies(df, columns=['Env'])
    
    # Separate baseline and task data
    baseline_df = df[df['N'] == 0].copy()
    task_df = df[df['N'] != 0].copy()
    
    if baseline_df.empty:
        print(f"No baseline data found in {file_path}")
        return pd.DataFrame(), pd.DataFrame()

    # Calculate baseline-corrected values
    for metric in ['FTCLI', 'FTPACLI', 'EngI']:
        task_df[f'{metric}_corrected'] = task_df.apply(
            lambda row: _calculate_corrected_value(row, baseline_df, metric),
            axis=1
        )
    
    # Encode task difficulty: 2-back as 0, 3-back as 1
    task_df['N'] = task_df['N'].map({2: 0, 3: 1})
    
    # Select and order output columns
    output_columns = [
        'Subject', 'Env_Control', 'Env_Nature', 'Env_Urban', 'N', 'NasaCLI', 'FTCLI_corrected',
        'FTPACLI_corrected', 'EngI_corrected', 'F1', 'TotalPress',
        'IncorrNo', 'CorrRatio', 'RTCorrAv', 'RTIncorrAv'
    ]
    corrected_df = task_df[output_columns]
    
    # Prepare motion artifacts data (includes baseline)
    motion_data = df[['Subject', 'N', 'Env_Control', 'Env_Nature', 'Env_Urban', 'MotionArt']].copy()
    
    return corrected_df.sort_values(['Env_Control', 'Env_Nature', 'Env_Urban', 'N']).reset_index(drop=True), motion_data

def _calculate_corrected_value(row, baseline_df, column):
    """Calculate baseline-normalized metric values"""
    # Find the Env column in row that is 1
    env_cols = ['Env_Control', 'Env_Nature', 'Env_Urban']
    env_active = [col for col in env_cols if row[col] == 1]
    if len(env_active) == 1:
        env_col = env_active[0]
    else:
        raise ValueError("Row has ambiguous or no active Env encoding")
    
    try:
        # Filter baseline_df where the same Env column is 1 and Subject matches
        baseline = baseline_df.loc[
            (baseline_df[env_col] == 1) &
            (baseline_df['Subject'] == row['Subject']),
            column
        ].values[0]
        
        if baseline == 0:
            return np.nan
            
        return (row[column] - baseline) / baseline
        
    except (IndexError, KeyError):
        return np.nan

def _update_final_dataset(final_path, new_data):
    """Update main dataset with new entries using targeted composite keys"""
    # Define identity columns for composite key
    identity_columns = ['Subject', 'Env_Control', 'Env_Nature', 'Env_Urban', 'N']
    
    if final_path.exists():
        existing_df = pd.read_csv(final_path)
        
        # Create keys using only identity columns
        existing_df['_key'] = existing_df[identity_columns].astype(str).agg('|'.join, axis=1)
        new_data['_key'] = new_data[identity_columns].astype(str).agg('|'.join, axis=1)
        
        # Remove existing entries that match new data keys
        existing_df = existing_df[~existing_df['_key'].isin(new_data['_key'])]
        updated_df = pd.concat([existing_df.drop('_key', axis=1), 
                              new_data.drop('_key', axis=1)])
    else:
        updated_df = new_data
    
    updated_df.to_csv(final_path, index=False)
    return updated_df

if __name__ == "__main__":
    process_all_participants()