import pandas as pd

def process_nasatlx(nasatlx_stream, environment, nback):
    """Process NASA-TLX data and calculate Cognitive Load Index (CLI)"""
    nasatlx_data = nasatlx_stream['time_series'][0]
    
    # Create DataFrame with raw scores
    df = pd.DataFrame([nasatlx_data], columns=[
        "Mental", "Physical", "Temporal", 
        "Performance", "Effort", "Frustration"
    ])
    
    # Add metadata and calculate CLI
    df['Timestamp'] = nasatlx_stream['time_stamps'][-1]
    df['Environment'] = environment
    df['N'] = nback
    df['NasatlxCli'] = df[[
        'Mental', 'Physical', 'Temporal',
        'Performance', 'Effort', 'Frustration'
    ]].sum(axis=1)
    
    # Remove original TLX components
    df.drop(columns=[
        'Mental', 'Physical', 'Temporal',
        'Performance', 'Effort', 'Frustration'
    ], inplace=True)
    
    return df

def process_game_performance(stream):
    """Process game data to extract performance metrics and task timing"""
    # Extract channels and create DataFrame
    channels = [ch['label'][0] for ch in stream['info']['desc'][0]['channels'][0]['channel']]
    df = pd.DataFrame(stream['time_series'], columns=channels)
    df['Timestamp'] = stream['time_stamps']
    
    # Add Environment column from stream name
    df['Environment'] = stream['info']['name'][0]
    
    # Get precise task timestamps
    start_time = df[df['Started'] == 1.0]['Timestamp'].iloc[0]
    end_time = df[df['Finished'] == 1.0]['Timestamp'].iloc[0]
    
    # Clean the DataFrame
    clean_df = df[
        (df['Timestamp'] > start_time) & 
        (df['Timestamp'] < end_time)
    ].drop(columns=['Started', 'Finished']).reset_index(drop=True)
    
    return clean_df, start_time, end_time

def process_in_game_df(df):
    """Calculate comprehensive performance metrics from cleaned game data"""
    # Fixed parameters based on experimental design
    ACTUAL_POSITIVES = 6  # Total targets in the VR task
    
    # Confusion matrix calculations
    tp = df['AnswerCorrect'].sum()  # True positives (correct hits)
    fp = (df['AnswerCorrect'] == 0).sum()  # False positives (incorrect responses)
    fn = ACTUAL_POSITIVES - tp  # Missed targets
    
    # Precision/Recall with division guards
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    # Reaction time analysis
    rt_series = df['ReactionTime']
    avg_rt_correct = rt_series[df['AnswerCorrect'] == 1].mean()
    avg_rt_incorrect = rt_series[df['AnswerCorrect'] == 0].mean()

    # Create and return df with single row
    # Create DataFrame and fill NaN
    result_df = pd.DataFrame([{
        'N': df['N'].iloc[0],
        'Environment': df['Environment'].iloc[0],
        'F1Score': round(f1, 4),
        'AvgRtCorrect': round(avg_rt_correct, 4),
        'AvgRtIncorrect': round(avg_rt_incorrect, 4),
        'TotalResponses': len(df),
        'HitRate': f"{tp}/{ACTUAL_POSITIVES}"
    }]).fillna(0)  # Fill NaN with 0
    
    return result_df