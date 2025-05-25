from config.paths import get_paths
import configparser
import os
import re
import pandas as pd
import numpy as np
from data_processing import (
    read_xdf,
    preprocess_behavioral,
    preprocess_vr_motion,
    preprocess_eeg,
    analyze_eeg,
    update_participant_data
)

# Read settings.ini from the repo root
config = configparser.ConfigParser()
# Read settings.ini from the same directory as main.py
config_path = os.path.join(os.path.dirname(__file__), "settings.ini")
config.read(config_path)
input_file = config['INPUT']['input_file']

def main(input_file):       
    # Read and align all data streams
    streams = read_xdf.load_xdf(input_file)
    
    # Extract participant ID from file path
    participant_id = re.search(r'sub-(P\d+)', input_file).group(1)
    
    # Process game data with metadata
    clean_game_df, game_meta = preprocess_behavioral.process_game_performance(streams['game'])
    
     #Extract task start and end times
    task_start = game_meta['Started']
    task_end = game_meta['Finished']
    
    # Process metrics and handle empty data (control condition)
    processed_game_df = preprocess_behavioral.process_in_game_df(clean_game_df, game_meta)
    
    # Process NASA-TLX with context from cleaned game data
    processed_nasatlx_df = preprocess_behavioral.process_nasatlx(streams['nasatlx'])
    
    # Process VR motion data during task
    motion_df = preprocess_vr_motion.calculate_kinematics(streams['motion'], task_start, task_end, buffer=0.25)
    
    # Initialize EEG-related variables with NaN defaults / handle the case when eeg processing fails
    raw_clean = None
    eeg_indexes = {
        'CognitiveLoadIndex': np.nan,
        'EngagementIndex': np.nan
    }

    try:
        # Process EEG data (cropped to exact task period)
        raw_clean = preprocess_eeg.process_eeg(streams['eeg'], input_file, task_start, task_end)
        
        # Add motion artifact annotations to EEG    
        raw_clean = preprocess_eeg.flag_eeg_artifacts_from_motion(raw_clean, motion_df)
        
        # Analyze EEG and get cognitive indexes
        eeg_indexes = analyze_eeg.calculate_indexes(raw_clean)
        
    except Exception as e:
        print(f'EEG processing failed for {participant_id}: {str(e)}')
    
    # Create combined dataframe for saving
    combined_df = pd.concat([
        processed_game_df.drop(columns=['Environment', 'N', 'V']),
        processed_nasatlx_df.drop(columns=['Timestamp'])
        ], axis=1).assign(
        Environment=processed_game_df['Environment'].iloc[0],
        N=processed_game_df['N'].iloc[0],
        V=processed_game_df['V'].iloc[0],
        CognitiveLoadIndex=round(eeg_indexes['CognitiveLoadIndex'], 6),
        EngagementIndex=round(eeg_indexes['EngagementIndex'], 6),
        ParticipantID=participant_id
    )
    
    # Define the column order
    cols_order = [
        'ParticipantID', 'Environment', 'N', 'V', 'CognitiveLoadIndex', 'EngagementIndex'
    ] + [col for col in combined_df.columns if col not in [
        'ParticipantID', 'Environment', 'N', 'V', 'CognitiveLoadIndex', 'EngagementIndex'
    ]]
    
    # Reorder the columns
    combined_df = combined_df[cols_order]

    # Update the dataframe with new data and save it 
    updated_df = update_participant_data.update_participant_data(combined_df, participant_id)   
        
    return {
        'nasatlx': processed_nasatlx_df,
        'motion': motion_df,
        'eeg': raw_clean,
        'game': processed_game_df,
        'cumulative' : updated_df
    }

if __name__ == "__main__":
    # Run main with path to input file
    main(input_file)