import pandas as pd
import re
import numpy as np

def process_nasatlx(nasatlx_stream):
    """Process NASA-TLX data or return empty scores if missing"""
    # Define column names with NasatlxCli
    cols = [
        'NasaCLI', 'NasatlxMental', 'NasatlxPhysical', 'NasatlxTemporal',
        'NasatlxPerformance', 'NasatlxEffort', 'NasatlxFrustration', 'Timestamp'
    ]
    
    if nasatlx_stream is None:
        return pd.DataFrame([[np.nan] + [np.nan]*6 + [pd.NaT]], columns=cols)
    
    try:
        nasatlx_data = nasatlx_stream['time_series'][0]
        time_stamp = nasatlx_stream['time_stamps'][-1]
    except (KeyError, IndexError):
        return pd.DataFrame([[np.nan] + [np.nan]*6 + [pd.NaT]], columns=cols)

    # Create dataframe for raw scores (excluding NasatlxCli)
    raw_cols = cols[1:-1]  # all except NasatlxCli and Timestamp
    df = pd.DataFrame([nasatlx_data], columns=raw_cols)
    
    # Calculate NasatlxCli as sum of raw scores
    df['NasaCLI'] = df.sum(axis=1)
    
    # Add Timestamp column
    df['Timestamp'] = time_stamp
    
    # NasatlxCli column first
    return df[['NasaCLI'] + raw_cols + ['Timestamp']]

def process_game_performance(stream):
    """Process game data with empty dataframe handling"""
    # Extract base data
    channels = [ch['label'][0] for ch in stream['info']['desc'][0]['channels'][0]['channel']]
    full_df = pd.DataFrame(stream['time_series'], columns=channels)
    full_df['Timestamp'] = stream['time_stamps']
    
    # Extract metadata before filtering
    meta = {
        'Environment': stream['info']['name'][-1],
        'N': full_df[full_df['Finished'] == 1.0]['N'].iloc[0],
        'V': full_df[full_df['Finished'] == 1.0]['V'].iloc[0],
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
    
    # Renaming the Environment column in meta DF 
    meta['Environment'] = environment
    
    # Default values setup
    defaults = {
        'F1': np.nan,
        'RTCorrAv': np.nan,
        'RTIncorrAv': np.nan,
        'TotalPress': np.nan,
        'CorrRatio': np.nan,
        'IncorrNo' : np.nan        
    }
    
    # Remove 'Started' and 'Finished' from meta for the output dataframe
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
    precision = tp / (tp + fp) if (tp + fp) > 0 else np.nan
    recall = tp / (tp + fn) if (tp + fn) > 0 else np.nan
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else np.nan

    # Reaction time analysis - sets them to nan if there are no answers
    rt_series = clean_df['ReactionTime']
    avg_rt_correct = rt_series[clean_df['AnswerCorrect'] == 1].mean()
    avg_rt_incorrect = rt_series[clean_df['AnswerCorrect'] == 0].mean()
    
    # Create and return dataframe with single row
    # Create dataframe and fill NaN
    result_df = pd.DataFrame([{
        'N': clean_df['N'].iloc[0],
        'V': clean_df['V'].iloc[0],
        'Environment': environment,
        'F1': f1,
        'RTCorrAv': avg_rt_correct,
        'RTIncorrAv': avg_rt_incorrect,
        'TotalPress': len(clean_df),
        'CorrRatio': (tp / ACTUAL_POSITIVES),
        'IncorrNo' : fp
    }])
    
    return result_df