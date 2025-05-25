import os
import pandas as pd

def update_participant_data(df, participant_id):
    # Create results directory structure
    results_dir = os.path.join('results', participant_id)
    os.makedirs(results_dir, exist_ok=True)
    
    # Construct file path with participant ID
    file_path = os.path.join(results_dir, f'{participant_id}_data.csv')
    
    if os.path.exists(file_path):
        existing_df = pd.read_csv(file_path)
        
        # Create composite keys for duplicate checking
        existing_df['_key'] = (
            existing_df['ParticipantID'].astype(str) + '|' +
            existing_df['Environment'].astype(str) + '|' +
            existing_df['N'].astype(str)
        )
        
        df['_key'] = (
            df['ParticipantID'].astype(str) + '|' +
            df['Environment'].astype(str) + '|' +
            df['N'].astype(str)
        )
        
        # Remove existing entries that match new data's keys
        existing_clean = existing_df[~existing_df['_key'].isin(df['_key'])].drop(columns=['_key'])
        
        # Combine cleaned existing data with new data (new entries overwrite old ones)
        updated_df = pd.concat([existing_clean, df.drop(columns=['_key'])], ignore_index=True)
    else:
        updated_df = df.drop(columns=['_key'], errors='ignore')
    
    # Save updated data
    updated_df.to_csv(file_path, index=False)
    print(f"Data saved to: {file_path}")
    
    return updated_df