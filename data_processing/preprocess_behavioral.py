import pandas as pd
import re

def process_nasatlx(nasatlx_stream):
    """Process NASA-TLX data or return empty scores if missing"""
    # Define column names
    cols = [
        'NasatlxMental', 'NasatlxPhysical', 'NadatlxTemporal',
        'NasatlxPerformance', 'NasatlxEffort', 'NasatlxFrustration',
        'Timestamp', 'NasatlxCli'
    ]
    
    if nasatlx_stream is None:
        # Return zero-filled dataframe with correct columns
        return pd.DataFrame([[0]*6 + [pd.NaT, 0]], columns=cols)
    
    try:
        nasatlx_data = nasatlx_stream['time_series'][0]
        time_stamp = nasatlx_stream['time_stamps'][-1]
    except (KeyError, IndexError):
        return pd.DataFrame([[0]*6 + [pd.NaT, 0]], columns=cols)

    # Create DataFrame with raw scores
    df = pd.DataFrame([nasatlx_data], columns=cols[:6])
    
    # Add metadata and calculate CLI
    df['Timestamp'] = time_stamp
    df['NasatlxCli'] = df.iloc[:, :6].sum(axis=1)
     
    return df[cols]  # Ensure consistent column order

def process_game_performance(stream):
    """Process game data with empty dataframe handling"""
    # Extract base data
    channels = [ch['label'][0] for ch in stream['info']['desc'][0]['channels'][0]['channel']]
    full_df = pd.DataFrame(stream['time_series'], columns=channels)
    full_df['Timestamp'] = stream['time_stamps']
    
    # Extract metadata before filtering
    meta = {
        'Environment': stream['info']['name'][-1],
        'N': full_df['N'].iloc[-1],
        'V': full_df['V'].iloc[-1],
        'Started': full_df[full_df['Started'] == 1.0]['Timestamp'].iloc[0],
        'Finished': full_df[full_df['Finished'] == 1.0]['Timestamp'].iloc[0]
    }

    # Filter task period
    clean_df = full_df[
        (full_df['Timestamp'] > meta['Started']) & 
        (full_df['Timestamp'] < meta['Finished'])
    ].drop(columns=['Started', 'Finished']).reset_index(drop=True)

    return clean_df, meta

def process_in_game_df(clean_df, meta):
    """Calculate metrics with empty dataframe support"""
    
    # Rename environment
    # Full environment string
    environment = meta['Environment']

    # Extract the scene part (e.g., 'UrbanScene', 'NatureScene', 'ControlScene')
    match = re.search(r'Unity\.(\w+)', environment)
    if match:
        environment = match.group(1)
    else:
        print(f"Warning: Could not extract environment.")
        environment = "Unknown"

    # Map to simplified environment name
    mapping = {
        'UrbanScene': 'Urban',
        'NatureScene': 'Nature',
        'ControlScene': 'Control'
    }
    environment = mapping.get(environment, environment)
    
    # Renaming the Environemnt column in meta DF 
    meta['Environment'] = environment
    
    # Default values setup
    defaults = {
        'F1Score': 0.0,
        'AvgRtCorrect': 0.0,
        'AvgRtIncorrect': 0.0,
        'TotalResponses': 0,
        'HitRate': 0.0
    }
    
    # Remove 'Started' and 'Finished' from meta for the output DataFrame
    meta_no_markers = {k: v for k, v in meta.items() if k not in ['Started', 'Finished']}
    
    # Ensure to work for empty data (control condition)
    if clean_df.empty:
        return pd.DataFrame([{**meta_no_markers, **defaults}])
    
    # Fixed parameters based on experimental design
    ACTUAL_POSITIVES = 6  # Total targets in the VR task
    
    # Confusion matrix calculations
    tp = clean_df['AnswerCorrect'].sum()  # True positives (correct hits)
    fp = (clean_df['AnswerCorrect'] == 0).sum()  # False positives (incorrect responses)
    fn = ACTUAL_POSITIVES - tp  # Missed targets
    
    # Precision/Recall with division guards
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    # Reaction time analysis
    rt_series = clean_df['ReactionTime']
    avg_rt_correct = rt_series[clean_df['AnswerCorrect'] == 1].mean()
    avg_rt_incorrect = rt_series[clean_df['AnswerCorrect'] == 0].mean()
    
    # Create and return df with single row
    # Create DataFrame and fill NaN
    result_df = pd.DataFrame([{
        'N': clean_df['N'].iloc[0],
        'V': clean_df['V'].iloc[0],
        'Environment': environment,
        'F1Score': round(f1, 6),
        'AvgRtCorrect': round(avg_rt_correct, 6),
        'AvgRtIncorrect': round(avg_rt_incorrect, 6),
        'TotalResponses': len(clean_df),
        'HitRate': round(tp / ACTUAL_POSITIVES, 6)
    }]).fillna(0)  # Fill NaN with 0
    
    return result_df