import mne
from mne.preprocessing import ICA
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
import picard
from mne_icalabel import label_components
from mne import make_fixed_length_events
from config.paths import get_paths

def process_eeg(stream, input_filename, start_time, end_time, base_output_dir=None):
    """Process EEG data with intermediate saves at each step."""
    # Get all paths from config-enabled module
    paths = get_paths(input_filename, base_output_dir)
     
    # Create raw data
    raw = _create_raw(stream)
    
    # Convert absolute times to relative
    eeg_start_time = stream["time_stamps"][0]
    tmin = max(0, start_time - eeg_start_time)  # Ensure non-negative
    tmax = min(raw.times[-1], end_time - eeg_start_time)  # Don't exceed EEG duration
    
    # Crop raw and save eeg    
    raw.crop(tmin=tmin, tmax=tmax)
    raw.save(paths['raw_fif'], overwrite=True)
    # Plot and save raw .fif as PNG
    _safe_plot_psd(raw, paths['raw_fif'])
    
    # Basic processing (filtering) and save
    raw_filtered = _basic_filtering(raw)
    raw_filtered.save(paths['filtered_fif'], overwrite=True)
    _safe_plot_psd(raw_filtered, paths['filtered_fif'])
    
    # Apply Oversampled Temporal Projection (OTP) for dry electrodes
    raw_otp = mne.preprocessing.oversampled_temporal_projection(raw_filtered, duration=5.0, picks='eeg', verbose=True)
    raw_otp.save(paths['otp_fif'], overwrite=True)  # Add this to your paths
    _safe_plot_psd(raw_otp, paths['otp_fif'])
    
    # Bad channel detection to remove extreme artifacts and save
    bads = _detect_bad_channels(raw_otp, paths, min_peak=20e-6, max_peak=500e-6, pre_ica=True) # 20µV-dry electrodes higher baseline noise 500µV captures major movement/contact artifacts
    raw_otp.info['bads'] = bads
    raw_bads_dropped = raw_otp.drop_channels(raw_otp.info['bads'])
    raw_bads_dropped.save(paths['bad_channels_1_fif'], overwrite=True)
    _safe_plot_psd(raw_bads_dropped, paths['bad_channels_1_fif']) 

    # ICA and save
    raw_ica = _ica_cleaning(raw_bads_dropped, paths)
    raw_ica.save(paths['ica_fif'], overwrite=True)
    _safe_plot_psd(raw_ica, paths['ica_fif'])
        
    # Bad channel detection for the second time with stricter limits for residual noise
    bads = _detect_bad_channels(raw_ica, paths, min_peak=5e-6, max_peak=100e-6, pre_ica=False) # Catch flat channels and catch residual artifacts
    raw_ica.info['bads'] = bads
    raw_bads_dropped_2 = raw_ica.drop_channels(raw.info['bads'])
    raw_bads_dropped_2.save(paths['bad_channels_2_fif'], overwrite=True)
    _safe_plot_psd(raw_bads_dropped_2, paths['bad_channels_2_fif'])
    
    return raw_bads_dropped_2

def flag_eeg_artifacts_from_motion(raw, motion_df):
    """Annotate EEG with motion artifacts based on pre-buffered motion_df."""
    # Create copy to avoid modifying original dataframe
    motion_df = motion_df.copy()
    
    # Convert motion timestamps to EEG's time framework
    eeg_start = raw.first_time
    motion_df['Timestamp'] = motion_df['Timestamp'] - motion_df['Timestamp'].iloc[0] + eeg_start
    
    # Synchronization check
    motion_start = motion_df['Timestamp'].iloc[0]
    if abs(eeg_start - motion_start) > 0.1:
        raise ValueError(f"EEG/motion timestamps misaligned by {abs(eeg_start - motion_start):.2f}s")

    # Calculate sampling rate robust to gaps/dropouts
    time_diffs = np.diff(motion_df['Timestamp'].values)
    if np.median(time_diffs) <= 1e-6:
        raise ValueError("Motion data timestamps invalid (near-zero sampling rate)")
    sfreq = 1 / np.median(time_diffs)

    # Find contiguous artifact blocks with clamping
    artifact_mask = motion_df['MotionArtifact'].values
    timestamps = motion_df['Timestamp'].values
    max_eeg_time = raw.first_time + raw.times[-1]
    
    onsets, durations = [], []
    in_artifact = False
    
    for i in range(len(artifact_mask)):
        if artifact_mask[i] and not in_artifact:
            # Start of new artifact block (clamped to EEG start)
            start_time = max(timestamps[i], eeg_start)
            in_artifact = True
        elif not artifact_mask[i] and in_artifact:
            # End of artifact block (include current sample)
            end_time = timestamps[i-1] + (1/sfreq)
            end_time = min(end_time, max_eeg_time)
            
            # Only add if duration > 0
            if end_time > start_time:
                onsets.append(start_time)
                durations.append(end_time - start_time)
                
            in_artifact = False

    # Handle final artifact block
    if in_artifact:
        end_time = min(timestamps[-1] + (1/sfreq), max_eeg_time)
        if end_time > start_time:
            onsets.append(start_time)
            durations.append(end_time - start_time)

    # Clamp all annotations to EEG time range
    clamped_onsets = []
    clamped_durations = []
    for onset, duration in zip(onsets, durations):
        # Adjust onset to be ≥ EEG start
        adj_onset = max(onset, eeg_start)
        # Calculate maximum possible duration
        max_dur = max_eeg_time - adj_onset
        # Only keep annotations with positive duration
        if max_dur > 0:
            clamped_onsets.append(adj_onset - eeg_start)  # Convert to relative time
            clamped_durations.append(min(duration, max_dur))

    # Create and add annotations
    annotations = mne.Annotations(
        onset=clamped_onsets,
        duration=clamped_durations,
        description='bad_motion'
    )
    raw.set_annotations(raw.annotations + annotations)
    return raw

def _create_raw(stream):
    eeg_data = np.array(stream["time_series"]).T / 1e6  # μV→V
    sfreq = float(stream["info"]["nominal_srate"][0])
    ch_names = "AF3,AF4,F3,Fz,F4,P3,Pz,P4".split(',')[:eeg_data.shape[0]]
    
    # Create info with standard electrode positions
    info = mne.create_info(ch_names, sfreq, 'eeg')
    info.set_montage('standard_1005', match_case=False) # Enobio
    
    return mne.io.RawArray(eeg_data, info)

def _basic_filtering(raw):
    # Filtering
    raw.filter(2, 40, fir_design='firwin') # 2 while 1Hz high-pass may retain baseline wander from dry electrodes.
    raw.notch_filter(50)

    return raw

def _detect_bad_channels(raw, paths, min_peak, max_peak, pre_ica):
    """Bad channel detection optimized for Enobio 8 dry electrodes"""
    # Ensure montage is set
    if not raw.info['dig']:
        raw.set_montage('standard_1020', match_case=False, on_missing='warn')
          
    # 1. Calculate PTP values
    try:
        ptp_values = _calculate_ptp_percentile(raw, pre_ica)
    except ValueError as e:
        raise RuntimeError(f"PTP calculation failed: {str(e)}")

    # 2. Identify bad channels using ptp_values
    bad_by_ptp = [
        raw.ch_names[i] 
        for i, ptp in enumerate(ptp_values)
        if (ptp < min_peak) or (ptp > max_peak)
    ]
    
    # 3. Final validation
    good_channels = [ch for ch in raw.ch_names if ch not in bad_by_ptp]
    has_frontal = any(ch.startswith('F') for ch in good_channels)
    has_parietal = any(ch.startswith('P') for ch in good_channels)
    
    # 4. Prepare detailed results
    details = {
        "bad_by_ptp": bad_by_ptp,
        "ptp_values": {ch: f"{ptp:.2e} V" for ch, ptp in zip(raw.ch_names, ptp_values)},
        "frontal_remaining": has_frontal,
        "parietal_remaining": has_parietal
    }
    
    # 5. Build log content
    log_content = (
        "Bad Channel Report\n"
        "---------------------------\n"
        f"- Bad by peak-to-peak amplitude: {details['bad_by_ptp']}\n"
        f"- Frontal channels remaining: {has_frontal}\n"
        f"- Parietal channels remaining: {has_parietal}\n"
        f"- PTP values:\n"
    ) + "\n".join([f"  {ch}: {val}" for ch, val in details['ptp_values'].items()])
    
    # Add critical error if missing regions
    if not (has_frontal and has_parietal):
        log_content += "\nERROR: Insufficient channels for index calculation - "
        log_content += "\nrequires at least one frontal and one parietal channel!"
    
    if pre_ica:
        with open(paths['bad_channels_log_1'], 'w') as f:        
            f.write(log_content)
    else:
        with open(paths['bad_channels_log_2'], 'w') as f:        
            f.write(log_content)
    
    return bad_by_ptp

def _ica_cleaning(raw, paths):
    """Perform automated ICA artifact rejection for eye blinks."""
    # Ensure figures directory exists
    os.makedirs(paths['figures'], exist_ok=True)
    
    # After dropping bad channels, you adjust n_components to match the new channel count
    current_channels = len(raw.ch_names) - len(raw.info['bads'])
    # Initialize and fit ICA
    ica = ICA(
        n_components=current_channels-1, # Rank= No. channels - 1
        method='picard',
        max_iter=1500,
        random_state=42,
        fit_params=dict(ortho=False)
    )
    ica.fit(raw)

    # Create component plots directory
    components_dir = os.path.join(paths['figures'], 'ica_components')
    os.makedirs(components_dir, exist_ok=True)

    # Generate and save individual component plots
    print("\nGenerating component detail plots...")
    for comp in range(ica.n_components_):
        fig_comp = ica.plot_properties(raw, picks=comp, show=False)
        fig_comp[0].savefig(os.path.join(components_dir, f'component_{comp:02d}.png'),
                            bbox_inches='tight')
        plt.close(fig_comp[0])
    print("Component details saved to:", components_dir)

    # Automated Exclusion---------------
    auto_excluded = []
    ic_labels = None
    try:
        ic_labels = label_components(raw, ica, method="iclabel")
        
        # Automatically exclude ALL eye components regardless of confidence
        auto_excluded = [
            idx for idx, label in enumerate(ic_labels['labels'])
            if ("eye" in label)
        ]
        
    except Exception as e:
        print(f"ICLabel Error: {str(e)} - Continuing without component exclusion")
    
    ica.exclude = auto_excluded

    # Reporting Phase---------------
    fig = plt.figure(figsize=(14, 8), dpi=100)
    gs = gridspec.GridSpec(nrows=3, ncols=4, height_ratios=[2, 2, 1])

    # Plot components in grid
    for i in range(min(ica.n_components_, 8)):
        ax = fig.add_subplot(gs[i // 4, i % 4])
        ica.plot_components(picks=i, axes=ax, show=False, res=32)
        if i in ica.exclude:
            ax.text(0.5, 0.5, 'X', color='red', fontsize=50, 
                    ha='center', va='center', transform=ax.transAxes, 
                    weight='bold')
        ax.set_title(f"ICA Component {i}", fontsize=9)

    # Report text
    ax_text = fig.add_subplot(gs[2, :])
    ax_text.axis('off')

    var_explained = ica.get_explained_variance_ratio(
        raw, 
        components=[c for c in range(ica.n_components_) if c not in ica.exclude]
    )['eeg']

    report_text = [
        "ICA ARTIFACT REJECTION REPORT",
        "============================",
        f"Excluded components: {ica.exclude}",
        f"Variance retained: {var_explained:.1%}",
        ""
    ]

    if ic_labels is not None:
        report_text.append("ICLabel classifications:")
        for comp in range(ica.n_components_):
            label = ic_labels['labels'][comp]
            prob = ic_labels['y_pred_proba'][comp]
            report_text.append(
                f"Component {comp}: {label} ({prob:.0%} confidence)"
            )
    else:
        report_text.append("ICLabel classification unavailable")

    ax_text.text(
        0.01, 0.99, "\n".join(report_text),
        fontfamily='monospace',
        fontsize=8,
        linespacing=1.2,
        verticalalignment='top',
        transform=ax_text.transAxes,
        bbox=dict(facecolor='white', alpha=0.95)
    )

    # Save and close
    save_path = os.path.join(paths['figures'], 'ica_report.png')
    fig.savefig(save_path, bbox_inches='tight')
    plt.close(fig)

    # Save log
    with open(paths['ica_rejection_log'], 'w') as f:
        f.write("\n".join(report_text))
        if ica.exclude:
            f.write("\n\nExclusion rationale: Automatic eye component removal")

    return ica.apply(raw)


def _calculate_ptp_percentile(raw, pre_ica):
    """Calculate specified percentile of PTP in sliding windows"""
    # Lenient bad channels identifier uses 90 percentile, while strict 95 percentile. 
    if pre_ica:
        percentile = 90
    else:
        percentile = 95
    
    # Dynamic windowing with validated overlap
    sfreq = raw.info['sfreq']
    
    # Calculate adaptive window parameters
    max_window_duration = 1.0  # Maximum 1s window
    min_samples = 5  # Minimum 5 samples per window
    
    # Calculate in samples
    n_per_seg = min(int(max_window_duration * sfreq), len(raw.times))
    n_per_seg = max(n_per_seg, min_samples)  # Ensure minimum
    
    # Calculate window duration in seconds
    window_duration = n_per_seg / sfreq  # NEW: Convert samples to time
    
    # 2. Calculate overlap (25% of window length)
    n_overlap = int(0.25 * n_per_seg)
    n_overlap = min(n_overlap, n_per_seg - 1)  # Critical fix
    
    # 3. Create events with validated parameters
    events = make_fixed_length_events(
        raw, 
        duration=window_duration,  # Use calculated duration
        overlap=n_overlap/sfreq
    )
    
    # Create epochs
    epochs = mne.Epochs(
        raw,
        events,
        tmin=0,
        tmax=window_duration,  # Use calculated duration
        baseline=None,
        preload=True,
        verbose=False
    )
    
    # Check if any epochs were created
    if len(epochs) == 0:
        raise ValueError("No epochs created - check event generation")
    
    # Calculate PTP
    ptp_matrix = np.ptp(epochs.get_data(), axis=2)  # Shape: (epochs, channels)
    return np.percentile(ptp_matrix, percentile, axis=0)

def _safe_plot_psd(raw, path):
        
        # Convert Path object to string for string replacement
        png_path = str(path).replace('.fif', '_psd.png')
        nperseg = min(512, raw.n_times)
        nperseg = max(nperseg, 5)  # Minimum 5 samples
        noverlap = min(int(0.25 * nperseg), nperseg - 1)
        
        if nperseg <= noverlap:
            print(f"Skipping PSD for {path}: invalid window params")
            return
            
        fig = raw.plot_psd(
            fmin=1, fmax=40,
            n_fft=nperseg,
            n_overlap=noverlap,
            show=False
        )
        fig.savefig(png_path, bbox_inches='tight')
        plt.close(fig)