import os
import pandas as pd

def save_participant_data(combined_df, participant_id):
      
    # Create results directory structure
    results_dir = os.path.join('results', participant_id)
    os.makedirs(results_dir, exist_ok=True)
    
    # Construct file path with participant ID
    file_path = os.path.join(results_dir, f'{participant_id}_data.csv')
    
    # Append or create new data
    if os.path.exists(file_path):
        existing_df = pd.read_csv(file_path)
        updated_df = pd.concat([existing_df, combined_df], ignore_index=True)
    else:
        updated_df = combined_df
    
    # Save
    updated_df.to_csv(file_path, index=False)
    print(f"Data saved to: {file_path}")