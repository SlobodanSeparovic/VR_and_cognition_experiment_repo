import mne

def calculate_indices(raw):
    """Calculate indices directly from Raw object"""
    theta, alpha, beta = calculate_band_power(raw)
    return {
        'cognitive_load': theta / alpha,
        'engagement': beta / (theta + alpha)
    }

def calculate_band_power(raw):
    """Calculate frontal theta (4-8Hz), parietal alpha (8-13Hz) and beta (13-30Hz) power"""
    # Select frontal and parietal channels based on eeg setup
    frontal_channels = ['AF3', 'AF4', 'F3', 'Fz', 'F4']
    parietal_channels = ['P3', 'Pz', 'P4']
    all_channels = raw.info['ch_names']
    
    # Verify available channels
    frontal_chs = [ch for ch in frontal_channels if ch in all_channels]
    parietal_chs = [ch for ch in parietal_channels if ch in all_channels]
    
    # Calculate PSD using Welch's method
    psd_frontal = raw.copy().pick(frontal_chs).compute_psd(fmin=4, fmax=8, method='welch', n_fft=2048)
    psd_parietal = raw.copy().pick(parietal_chs).compute_psd(fmin=8, fmax=13, method='welch', n_fft=2048)
    psd_beta = raw.copy().pick(all_channels).compute_psd(fmin=13, fmax=30, method='welch', n_fft=2048)
    
    # Calculate absolute power (mean across channels then frequencies to reduce bias from dominant individual channels)
    theta = psd_frontal.get_data().mean(axis=0).mean()  # Frontal theta
    alpha = psd_parietal.get_data().mean(axis=0).mean()  # Parietal alpha
    beta = psd_beta.get_data().mean(axis=0).mean()  # Global beta
    return theta, alpha, beta
