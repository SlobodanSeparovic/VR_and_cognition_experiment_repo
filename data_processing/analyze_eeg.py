import mne

def calculate_indices(raw):
    """Calculate indices directly from Raw object"""
    theta, alpha, beta = calculate_band_power(raw)
    return {
        'cognitive_load': theta / alpha,
        'engagement': beta / (theta + alpha)
    }

def calculate_band_power(raw):
    """Calculate frontal theta (4-8Hz), parietal alpha (8-13Hz), and global beta (13-30Hz) power.
    Raises ValueError if required channels are missing."""
    
    # Define target channels
    frontal_channels = ['AF3', 'AF4', 'F3', 'Fz', 'F4']
    parietal_channels = ['P3', 'Pz', 'P4']
    available_channels = raw.info['ch_names']
    
    # Find available electrodes in each region
    frontal_chs = [ch for ch in frontal_channels if ch in available_channels]
    parietal_chs = [ch for ch in parietal_channels if ch in available_channels]

    # Validate channel presence
    missing_regions = []
    if not frontal_chs:
        missing_regions.append("frontal")
    if not parietal_chs:
        missing_regions.append("parietal")
    if missing_regions:
        raise ValueError(
            f"Cannot calculate band power - missing {', '.join(missing_regions)} channels. "
            "Participant data cannot be used for the experiment."
        )

    # Calculate power spectra
    psd_theta = raw.copy().pick(frontal_chs).compute_psd(fmin=4, fmax=8, method='welch', n_fft=2048)
    psd_alpha = raw.copy().pick(parietal_chs).compute_psd(fmin=8, fmax=13, method='welch', n_fft=2048)
    psd_beta = raw.copy().pick(available_channels).compute_psd(fmin=13, fmax=30, method='welch', n_fft=2048)

    # Calculate mean absolute power
    theta = psd_theta.get_data().mean(axis=0).mean()
    alpha = psd_alpha.get_data().mean(axis=0).mean()
    beta = psd_beta.get_data().mean(axis=0).mean()

    return theta, alpha, beta
