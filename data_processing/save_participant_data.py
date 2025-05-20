import os
import pandas as pd

def save_participant_data(combined_df, participant_id):
    # Create results directory structure
    results_dir = os.path.join('results', participant_id)
    os.makedirs(results_dir, exist_ok=True)
    
    # Construct file path with participant ID
    file_path = os.path.join(results_dir, f'{participant_id}_data.csv')
    
    if os.path.exists(file_path):
        existing_df = pd.read_csv(file_path)
        
        # Create composite key for duplicate checking
        existing_df['_key'] = (
            existing_df['ParticipantID'].astype(str) + '|' +
            existing_df['Environment'].astype(str) + '|' +
            existing_df['N'].astype(str)
        )
        
        combined_df['_key'] = (
            combined_df['ParticipantID'].astype(str) + '|' +
            combined_df['Environment'].astype(str) + '|' +
            combined_df['N'].astype(str)
        )
        
        # Filter out existing combinations
        new_data = combined_df[~combined_df['_key'].isin(existing_df['_key'])].drop(columns=['_key'])
        existing_df = existing_df.drop(columns=['_key'])
        
        # Concatenate only new entries
        updated_df = pd.concat([existing_df, new_data], ignore_index=True)
    else:
        updated_df = combined_df.drop(columns=['_key'], errors='ignore')
    
    # Save cleaned data
    updated_df.to_csv(file_path, index=False)
    print(f"Data saved to: {file_path}")
    
    return updated_df