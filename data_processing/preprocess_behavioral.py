import pandas as pd

def process_nasatlx(nasatlx_stream, environment):
    """Process NASA-TLX data with environment context"""
    nasatlx_data = nasatlx_stream['time_series'][0]
    df = pd.DataFrame([nasatlx_data], columns=["Mental", "Physical", "Temporal", 
                                              "Performance", "Effort", "Frustration"])
    df['Timestamp'] = nasatlx_stream['time_stamps'][-1]  # Use last timestamp
    df['Environment'] = environment
    return df

def process_game_performance(stream):
    channels = [ch['label'][0] for ch in stream['info']['desc'][0]['channels'][0]['channel']]
    df = pd.DataFrame(stream['time_series'], columns=channels)
    df['Timestamp'] = stream['time_stamps']
    return df[~df['Started'].isin([1.0]) & ~df['Finished'].isin([1.0])]
