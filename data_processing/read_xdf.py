import pyxdf
import pandas as pd

def load_xdf(fname):
    streams, _ = pyxdf.load_xdf(fname, synchronize_clocks=True)
    
    # Process game stream
    game_stream = next(s for s in streams if any(kw in s["info"]["name"][0].lower() for kw in ("urban", "nature", "control")))
    game_cols = [ch['label'][0] for ch in game_stream['info']['desc'][0]['channels'][0]['channel']]
    game_df = pd.DataFrame(game_stream['time_series'], columns=game_cols)
    game_df['Timestamp'] = game_stream['time_stamps']
    
    # Add Environment column from stream name
    game_df['Environment'] = game_stream['info']['name'][0]
    
    # Get task time window
    start_time = game_df[game_df['Started'] == 1.0]['Timestamp'].iloc[0]
    end_time = game_df[game_df['Finished'] == 1.0]['Timestamp'].iloc[0]
    
    # Process motion stream with timestamp alignment
    motion_stream = next(s for s in streams if "motion" in s["info"]["name"][0].lower())
    motion_cols = [
        'HmdPosX', 'HmdPosY', 'HmdPosZ', 'HmdRotX', 'HmdRotY', 'HmdRotZ', 'HmdRotW',
        'LeftPosX', 'LeftPosY', 'LeftPosZ', 'LeftRotX', 'LeftRotY', 'LeftRotZ', 'LeftRotW',
        'RightPosX', 'RightPosY', 'RightPosZ', 'RightRotX', 'RightRotY', 'RightRotZ', 'RightRotW',
        'LeftTrigger', 'RightTrigger'
    ]
    motion_df = pd.DataFrame(motion_stream["time_series"], columns=motion_cols)
    motion_df['Timestamp'] = motion_stream["time_stamps"]
    motion_df = motion_df[(motion_df.Timestamp >= start_time) & (motion_df.Timestamp <= end_time)]
    
    return {
        'eeg': next(s for s in streams if "eeg" in s["info"]["name"][0].lower()),
        'game': game_df,
        'nasatlx': next(s for s in streams if s["info"]["type"][0] == "Responses"),
        'motion': motion_df  # Pre-cropped to task period with Timestamp
    }