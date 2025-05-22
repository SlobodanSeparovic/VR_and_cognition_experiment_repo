import mne

def calculate_indices(raw):
    """Calculate indices directly from Raw object"""
    frontal_theta, parietal_alpha, global_theta, global_alpha, global_beta = _calculate_band_power(raw)
    return {
        'cognitive_load': frontal_theta / parietal_alpha,
        'engagement': global_beta / (global_theta + global_alpha)
    }

def _calculate_band_power(raw):
    """Calculate band power indices with robust Welch parameters"""
    sfreq = raw.info['sfreq']
    
    # Dynamic window sizing
    max_window_sec = 2.0  # Maximum window length
    min_samples = int(0.5 * sfreq)  # Minimum 0.5s of data
    
    # Calculate in samples
    n_per_seg = min(int(max_window_sec * sfreq), raw.n_times)
    n_per_seg = max(n_per_seg, min_samples)
    n_overlap = int(0.25 * n_per_seg)  # 25% overlap
    n_overlap = min(n_overlap, n_per_seg - 1)  # Critical clamp
    
    # Frequency bands
    bands = {
        'theta': (4, 8),
        'alpha': (8, 13),
        'beta': (13, 31)
    }
    
    # Channel groups
    frontal_chs = [ch for ch in raw.ch_names if ch.startswith('F')]
    parietal_chs = [ch for ch in raw.ch_names if ch.startswith('P')]
    
    # Calculate power with validated parameters
    def _safe_psd(chs, fmin, fmax):
        if len(chs) == 0:
            return 0
        return raw.copy().pick(chs).compute_psd(
            fmin=fmin, fmax=fmax,
            method='welch',
            n_fft=n_per_seg,
            n_per_seg=n_per_seg,
            n_overlap=n_overlap,
            reject_by_annotation=True
        ).get_data().mean()
    
    return (
        _safe_psd(frontal_chs, *bands['theta']),  # Frontal theta
        _safe_psd(parietal_chs, *bands['alpha']), # Parietal alpha
        _safe_psd(raw.ch_names, *bands['theta']), # Global theta
        _safe_psd(raw.ch_names, *bands['alpha']), # Global alpha
        _safe_psd(raw.ch_names, *bands['beta'])   # Global beta
    )
