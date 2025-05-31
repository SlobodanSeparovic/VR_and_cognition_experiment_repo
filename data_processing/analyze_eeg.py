import mne

def calculate_indexes(raw):
    """Calculate indexes and artifact ratio for reporting"""
    frontal_theta, parietal_alpha, global_theta, global_alpha, global_beta, artifact_ratio = _calculate_band_power(raw)
    return {
        'FTCLI': frontal_theta,
        'FTPACLI': frontal_theta / parietal_alpha,
        'EngI': global_beta / (global_theta + global_alpha),
        'ArtRatio' : artifact_ratio
    }

def _calculate_band_power(raw):
    """Calculate band power while excluding annotated artifact periods.
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
    
    # Get frequency for proper binning of the data and ratio of data flagged as bad
    sfreq = int(raw.info['sfreq'])
    
    # Calculate artifact percentage including short segments
    bad_samples = sum(int(ann['duration'] * sfreq) for ann in raw.annotations if ann['description'] in ['bad_motion_artifact', 'bad_short_segment'])
    artifact_ratio = (bad_samples / raw.n_times) if raw.n_times > 0 else 0.0
        
    # Calculate power spectra while rejecting the motion artifact periods
    psd_frontal_theta = raw.copy().pick(frontal_chs).compute_psd(fmin=4, fmax=7, n_fft= sfreq, method='welch', reject_by_annotation=True)
    psd_parietal_alpha = raw.copy().pick(parietal_chs).compute_psd(fmin=8, fmax=13, n_fft=sfreq, method='welch', reject_by_annotation=True)
    psd_global_theta = raw.copy().pick(available_channels).compute_psd(fmin=4, fmax=7, n_fft=sfreq, method='welch', reject_by_annotation=True)
    psd_global_alpha = raw.copy().pick(available_channels).compute_psd(fmin=8, fmax=13, n_fft=sfreq, method='welch', reject_by_annotation=True)
    psd_global_beta = raw.copy().pick(available_channels).compute_psd(fmin=14, fmax=30, n_fft=sfreq, method='welch', reject_by_annotation=True)
    
    # Check bin frequencies:
    # For frontal theta (4-7 Hz)
    print("Frontal theta bin frequencies:", psd_frontal_theta.freqs)
    # For parietal alpha (8-13 Hz) 
    print("Parietal alpha bin frequencies:", psd_parietal_alpha.freqs)    
    # For global theta (5-7 Hz)
    print("Global theta bin frequencies:", psd_global_theta.freqs)
    # For global alpha (8-13 Hz) 
    print("Parietal alpha bin frequencies:", psd_global_alpha.freqs)    
    # For global beta (14-30 Hz) 
    print("Beta bin frequencies:", psd_global_beta.freqs)    
      
    # Calculate mean absolute power
    frontal_theta = psd_frontal_theta.get_data().mean(axis=0).mean()
    parietal_alpha = psd_parietal_alpha.get_data().mean(axis=0).mean()
    global_theta = psd_global_theta.get_data().mean(axis=0).mean()
    global_alpha = psd_global_alpha.get_data().mean(axis=0).mean()
    global_beta = psd_global_beta.get_data().mean(axis=0).mean()

    return frontal_theta, parietal_alpha, global_theta, global_alpha, global_beta, artifact_ratio