from config.paths import get_paths
from data_processing import (
    read_xdf,
    preprocess_behavioral,
    preprocess_vr_motion,
    preprocess_eeg,
    analyze_eeg
)

def main(input_file):
    paths = get_paths(input_file)
    
    # 1. Read data (time-aligned and timestamped)
    streams = read_xdf.load_xdf(input_file)
    
    # 2. Process behavioral data
    nasatlx_df = preprocess_behavioral.process_nasatlx(
        streams['nasatlx'], 
        streams['game']['Environment'].iloc[0]  # Get from game DataFrame
    )
    
    ### Process VR motion (use pre-cropped data)
    motion_df = preprocess_vr_motion.calculate_kinematics(streams['motion'])
    
    ### Process EEG (with task-period cropping)
    # Get task timestamps from game data
    start_time = streams['game']['Timestamp'].iloc[0]
    end_time = streams['game']['Timestamp'].iloc[-1]

    # Process EEG and get Raw object
    raw_clean = preprocess_eeg.process_eeg(streams['eeg'], paths, start_time, end_time)

    # Analyze directly from Raw object
    indices = analyze_eeg.calculate_indices(raw_clean)
    
    print(f"Cognitive Load Index: {indices['cognitive_load']:.2f}")
    print(f"Engagement Index: {indices['engagement']:.2f}")

if __name__ == "__main__":
    input_file = r"D:\master_thesis\data_files\P011\sub-P011_ses-S001_task-Default_run-001_eeg.xdf"
    main(input_file)