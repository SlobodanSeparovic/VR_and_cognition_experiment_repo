import os
import pandas as pd
import numpy as np
from pathlib import Path

def process_all_participants(results_dir='results'):
    """Main function to process all participant data with new format"""
    final_data = []
    motion_artifacts_data = []

    for participant_dir in Path(results_dir).iterdir():
        if not participant_dir.is_dir():
            continue
            
        participant_id = participant_dir.name
        input_file = participant_dir / f"{participant_id}_data.csv"
        output_file = participant_dir / f"{participant_id}_data_corrected.csv"

        if not input_file.exists():
            continue

        # Process participant data and get motion artifacts
        corrected_df, motion_artifacts_df = _process_participant_file(input_file)
        if corrected_df.empty:
            continue

        # Save corrected data
        corrected_df.to_csv(output_file, index=False)
        print(f"Saved corrected data for {participant_id} to {output_file}")

        # Collect motion artifacts and final data
        motion_artifacts_data.append(motion_artifacts_df)
        final_data.append(corrected_df)

    # Update final datasets
    if final_data:
        final_df = pd.concat(final_data, ignore_index=True)
        final_path = Path(results_dir) / 'final_dataset.csv'
        
        # Define the desired column order for the final dataset
        desired_order = ['Subject', 'Env', 'N', 'NasaCLI', 'FTCLI_corrected', 'FTPACLI_corrected', 'EngI_corrected',
                         'F1', 'TotalPress', 'IncorrNo', 'CorrRatio', 'RTCorrAv', 'RTIncorrAv']

        # Reorder columns in the final_df
        final_df = final_df[[col for col in desired_order if col in final_df.columns]]
        _update_final_dataset(final_path, final_df)

        # Handle motion artifacts
        motion_artifacts_df = pd.concat(motion_artifacts_data, ignore_index=True)
        motion_path = Path(results_dir) / 'motion_artifacts.csv'
        _update_final_dataset(motion_path, motion_artifacts_df)

        # Fill NaN values in final dataset
        _fill_nan_with_group_means(final_path)
        
        print(f"Final dataset updated at {final_path}")
        print(f"Motion artifacts dataset updated at {motion_path}")
    else:
        print("No participant data found for processing")

def _process_participant_file(file_path):
    """Process individual participant file with new format"""
    df = pd.read_csv(file_path)
    
    # Convert Subject to integer
    df['Subject'] = df['Subject'].str.replace('P', '').astype(int)
    
    # Encode Environment numerically
    env_encoding = {'Control': 0, 'Nature': 1, 'Urban': 2}
    df['Env'] = df['Env'].map(env_encoding)
    
    # Split baseline (N=0) and task data (N>0)
    baseline_df = df[df['N'] == 0].copy()
    task_df = df[df['N'] != 0].copy()
    
    if baseline_df.empty:
        print(f"No baseline data found in {file_path}")
        return pd.DataFrame(), pd.DataFrame()

    # Calculate corrected indexes for FTCLI, FTPACLI, and EngI
    for col in ['FTCLI', 'FTPACLI', 'EngI']:
        task_df[f'{col}_corrected'] = task_df.apply(
            lambda row: _calculate_corrected_index(row, baseline_df, col),
            axis=1
        )
    
    # Prepare corrected dataframe
    corrected_df = task_df.drop(columns=['V', 'FTCLI', 'FTPACLI', 'EngI', 'MotionArt'])
       
    # Order of columns for output
    desired_order = ['Subject', 'Env', 'N', 'NasaCLI', 'FTCLI_corrected', 'FTPACLI_corrected', 'EngI_corrected', 
                     'F1', 'TotalPress', 'IncorrNo', 'CorrRatio', 'RTCorrAv', 'RTIncorrAv']
    
    # Reorder columns (keep only those that exist in the DataFrame)
    corrected_df = corrected_df[[col for col in desired_order if col in corrected_df.columns]]
    
    # Prepare motion artifacts data (includes N=0)
    motion_artifacts_df = df[['Subject', 'N', 'Env', 'MotionArt']].copy()
    
    # Sort by Environment and N
    return corrected_df.sort_values(['Env', 'N']).reset_index(drop=True), motion_artifacts_df

def _calculate_corrected_index(row, baseline_df, column):
    """Calculate baseline-corrected values using formula: (value - baseline)/baseline"""
    try:
        baseline = baseline_df.loc[
            (baseline_df['Env'] == row['Env']) &
            (baseline_df['Subject'] == row['Subject']),
            column
        ].values[0]
        return (row[column] - baseline) / baseline if baseline != 0 else np.nan
    except (IndexError, KeyError):
        return np.nan

def _update_final_dataset(final_path, new_data):
    """Update final dataset with conflict resolution"""
    if final_path.exists():
        existing_df = pd.read_csv(final_path)
        # Remove existing entries that match new data
        composite_key = new_data.astype(str).apply('|'.join, axis=1)
        existing_df['_key'] = existing_df.astype(str).apply('|'.join, axis=1)
        existing_df = existing_df[~existing_df['_key'].isin(composite_key)]
        updated_df = pd.concat([existing_df.drop('_key', axis=1), new_data])
    else:
        updated_df = new_data
    
    updated_df.to_csv(final_path, index=False)
    return updated_df

def _fill_nan_with_group_means(final_path):
    """Fill missing values using group averages"""
    df = pd.read_csv(final_path)
    
    # Group by Environment and N
    grouped = df.groupby(['Env', 'N'])
    
    # Fill NaN values for reaction time metrics
    df['RTCorrAv'] = grouped['RTCorrAv'].transform(lambda x: x.fillna(x.mean()))
    df['RTIncorrAv'] = grouped['RTIncorrAv'].transform(lambda x: x.fillna(x.mean()))
    
    df.to_csv(final_path, index=False)


if __name__ == "__main__":
    process_all_participants()
