import os
import pandas as pd

def update_participant_data(df, participant_id):
    """Create csv file per participant or update if it already exists with new data"""
    # Create results directory structure
    results_dir = os.path.join('results', participant_id)
    os.makedirs(results_dir, exist_ok=True)
    
    # Construct file path with participant ID
    file_path = os.path.join(results_dir, f'{participant_id}_data.csv')
    
    if os.path.exists(file_path):
        existing_df = pd.read_csv(file_path)
        
        # Create composite keys for duplicate checking
        existing_df['_key'] = (
            existing_df['Subject'].astype(str) + '|' +
            existing_df['Env'].astype(str) + '|' +
            existing_df['N'].astype(str)
        )
        
        df['_key'] = (
            df['Subject'].astype(str) + '|' +
            df['Env'].astype(str) + '|' +
            df['N'].astype(str)
        )
        
        # Remove existing entries that match new data's keys, drop them and handle empty columns
        existing_clean = existing_df[~existing_df['_key'].isin(df['_key'])].drop(columns=['_key'])
        empty_cols_existing = existing_clean.columns[existing_clean.isna().all()].tolist()  # Identify empty columns
        existing_clean = existing_clean.drop(columns=empty_cols_existing)
        
        
        # Remove '_key' and empty columns from the new DataFrame (new entries overwrite old ones)
        clean_df = df.drop(columns=['_key'])
        empty_cols_clean = clean_df.columns[clean_df.isna().all()].tolist()  # Identify empty columns
        clean_df = clean_df.drop(columns=empty_cols_clean)

        # Concatenate cleaned existing data with new data
        updated_df = pd.concat([existing_clean, clean_df], ignore_index=True)  
              
    else:
        updated_df = df.drop(columns=['_key'], errors='ignore')
    
    # Save updated data
    updated_df.to_csv(file_path, index=False)
    print(f"Data saved to: {file_path}")
    
    return updated_df