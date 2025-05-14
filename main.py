from config.paths import get_paths
import configparser
import os
from data_processing import (
    read_xdf,
    preprocess_behavioral,
    preprocess_vr_motion,
    preprocess_eeg,
    analyze_eeg
)

# Read settings.ini from the repo root
config = configparser.ConfigParser()
# Read settings.ini from the same directory as main.py
config_path = os.path.join(os.path.dirname(__file__), "settings.ini")
config.read(config_path)


def main(input_file):      
    # Get paths
    paths = get_paths(input_file)
    
    # Read and align all data streams
    streams = read_xdf.load_xdf(input_file)
    
    # Process game data to get precise task timing and clean DF
    clean_game_df, task_start, task_end = preprocess_behavioral.process_game_performance(streams['game'])
    
    # Process clean_game_df to get relevant metrics
    processed_game_df = preprocess_behavioral.process_in_game_df(clean_game_df)
    
    # Process NASA-TLX with context from cleaned game data
    processed_nasatlx_df = preprocess_behavioral.process_nasatlx(
        streams['nasatlx'],
        processed_game_df['Environment'].iloc[0],
        processed_game_df['N'].iloc[0]
    )
    
        
    # Process VR motion data during task, read buffer fom config and pass it
    motion_df = preprocess_vr_motion.calculate_kinematics(
        streams['motion'],
        task_start,
        task_end,
        buffer=0.2
    )
    
    # Process EEG data (cropped to exact task period)
    raw_clean = preprocess_eeg.process_eeg(
        streams['eeg'],
        paths,
        task_start,
        task_end
    )
    
    # Add motion artifact annotations to EEG
    raw_clean = preprocess_eeg.flag_eeg_artifacts_from_motion(raw_clean, motion_df)
    
    # Analyze EEG and get cognitive indices
    indices = analyze_eeg.calculate_indices(raw_clean)
    
    # Print results with aligned metrics
    print(f"\nTask Duration: {task_end - task_start:.2f}s")
    print(f"Cognitive Load Index: {indices['cognitive_load']:.2f}")
    print(f"Engagement Index: {indices['engagement']:.2f}")
    print(f"F1 Score: {processed_game_df['F1Score'].iloc[0]:.2f}")
    
    return {
        'nasatlx': processed_nasatlx_df,
        'motion': motion_df,
        'eeg': raw_clean,
        'game': processed_game_df
    }

if __name__ == "__main__":
    # Run main with path to input file from congfig file
    main(config['INPUT']['input_file'])