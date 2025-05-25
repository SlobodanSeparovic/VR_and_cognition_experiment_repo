import os
import pandas as pd
import numpy as np
from pathlib import Path

def process_all_participants(results_dir='results'):
    """Main function to process all participant data"""
    final_data = []
    
    for participant_dir in Path(results_dir).iterdir():
        if not participant_dir.is_dir():
            continue
            
        participant_id = participant_dir.name
        input_file = participant_dir / f"{participant_id}_data.csv"
        output_file = participant_dir / f"{participant_id}_data_corrected.csv"
        
        if not input_file.exists():
            continue
            
        # Process individual participant data
        corrected_df = _process_participant_file(input_file)
        if corrected_df.empty:
            continue
            
        # Save corrected data
        corrected_df.to_csv(output_file, index=False)
        print(f"Saved corrected data for {participant_id} to {output_file}")
        
        # Prepare for final dataset
        final_data.append(corrected_df)
    
    # Update final aggregated dataset
    if final_data:
        final_df = pd.concat(final_data, ignore_index=True)
        final_path = Path(results_dir) / 'final_dataset.csv'
        _update_final_dataset(final_path, final_df)
        print(f"Final dataset updated at {final_path}")
    else:
        print("No participant data found for processing")


def _calculate_corrected_index(row, baseline_df, index_name):
    """Calculate corrected index value using baseline"""
    try:
        baseline = baseline_df.loc[
            (baseline_df['Environment'] == row['Environment']) &
            (baseline_df['ParticipantID'] == row['ParticipantID']),
            index_name
        ].values[0]
        return row[index_name] / baseline if not np.isnan(baseline) and baseline != 0 else np.nan
    except (IndexError, KeyError):
        return np.nan

def _process_participant_file(file_path):
    """Process individual participant file and return corrected dataframe"""
    df = pd.read_csv(file_path)
    
    # Split into baseline and task data
    baseline_df = df[df['N'] == 0].copy()
    task_df = df[df['N'] != 0].copy()
    
    if baseline_df.empty:
        print(f"No baseline data found in {file_path}")
        return pd.DataFrame()

    # Calculate corrected indices
    for index in ['CognitiveLoadIndex', 'EngagementIndex']:
        task_df[index + '_corrected'] = task_df.apply(
            lambda row: _calculate_corrected_index(row, baseline_df, index),
            axis=1
        )
    
    # Drop original indices and rename corrected columns
    task_df = task_df.drop(columns=['CognitiveLoadIndex', 'EngagementIndex', 'V'])
    task_df = task_df.rename(columns={
        'CognitiveLoadIndex_corrected': 'CognitiveLoadIndex',
        'EngagementIndex_corrected': 'EngagementIndex'
    })
    
    # Add average rows per environment
    avg_rows = []
    for env in task_df['Environment'].unique():
        env_df = task_df[task_df['Environment'] == env]
        avg_row = env_df.select_dtypes(include=np.number).mean(numeric_only=True)
        avg_row['ParticipantID'] = env_df['ParticipantID'].iloc[0]
        avg_row['Environment'] = env
        avg_row['N'] = 'Average'
        avg_rows.append(avg_row)
    
    corrected_df = pd.concat([task_df, pd.DataFrame(avg_rows)], ignore_index=True)
    return corrected_df

def _update_final_dataset(final_path, new_data):
    """Update final aggregated dataset with new data"""
    if os.path.exists(final_path):
        final_df = pd.read_csv(final_path)
        
        # Remove existing entries that match new_data
        composite_keys = new_data[['ParticipantID', 'Environment', 'N']].astype(str).apply('|'.join, axis=1)
        final_df['_key'] = final_df[['ParticipantID', 'Environment', 'N']].astype(str).apply('|'.join, axis=1)
        final_df = final_df[~final_df['_key'].isin(composite_keys)]
        
        # Combine old and new data
        final_df = pd.concat([final_df.drop(columns=['_key']), new_data], ignore_index=True)
    else:
        final_df = new_data
    
    final_df.to_csv(final_path, index=False)
    return final_df


if __name__ == "__main__":
    process_all_participants()
