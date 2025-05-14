import mne

def calculate_indices(raw):
    """Calculate indices directly from Raw object"""
    frontal_theta, parietal_alpha, global_theta, global_alpha, global_beta = _calculate_band_power(raw)
    return {
        'cognitive_load': frontal_theta / parietal_alpha,
        'engagement': global_beta / (global_theta + global_alpha)
    }

def _calculate_band_power(raw):
    """Calculate frontal theta (4-7Hz), parietal alpha (8-12Hz), and global beta (13-30Hz) power.
    Raises ValueError if required channels are missing.
    Calculate band power while excluding annotated artifact periods."""
    
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

    n_fft = 512  # FFT length (samples)
    n_per_seg = 512  # Segment length (samples)
    n_overlap = int(0.75 * n_per_seg)  # 75% overlap
    
    # Calculate power spectra while rejecting the motion artifact periods
    psd_frontal_theta = raw.copy().pick(frontal_chs).compute_psd(fmin=4, fmax=8, method='welch', n_fft=n_fft, n_per_seg=n_per_seg, n_overlap=n_overlap, reject_by_annotation=True)
    psd_parietal_alpha = raw.copy().pick(parietal_chs).compute_psd(fmin=8, fmax=13, method='welch', n_fft=n_fft, n_per_seg=n_per_seg, n_overlap=n_overlap, reject_by_annotation=True)
    psd_global_theta = raw.copy().pick(available_channels).compute_psd(fmin=4, fmax=8, method='welch', n_fft=n_fft, n_per_seg=n_per_seg, n_overlap=n_overlap, reject_by_annotation=True)
    psd_global_alpha = raw.copy().pick(available_channels).compute_psd(fmin=8, fmax=13, method='welch', n_fft=n_fft, n_per_seg=n_per_seg, n_overlap=n_overlap, reject_by_annotation=True)
    psd_global_beta = raw.copy().pick(available_channels).compute_psd(fmin=13, fmax=31, method='welch', n_fft=n_fft, n_per_seg=n_per_seg, n_overlap=n_overlap, reject_by_annotation=True)

    # Calculate mean absolute power
    frontal_theta = psd_frontal_theta.get_data().mean(axis=0).mean()
    parietal_alpha = psd_parietal_alpha.get_data().mean(axis=0).mean()
    global_theta = psd_global_theta.get_data().mean(axis=0).mean()
    global_alpha = psd_global_alpha.get_data().mean(axis=0).mean()
    global_beta = psd_global_beta.get_data().mean(axis=0).mean()

    return frontal_theta, parietal_alpha, global_theta, global_alpha, global_beta
