import mne
from mne.preprocessing import ICA
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
from scipy import interpolate
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
    fig = raw.compute_psd().plot(show=False)
    png_path = str(paths['raw_fif']).replace('.fif', '_psd.png')
    fig.savefig(png_path, bbox_inches='tight')
    plt.close(fig)
    
    # Basic processing (filtering) and save
    raw_filtered = _basic_filtering(raw)
    raw_filtered.save(paths['filtered_fif'], overwrite=True)
    fig = raw_filtered.compute_psd().plot(show=False)
    png_path = str(paths['filtered_fif']).replace('.fif', '_psd.png')
    fig.savefig(png_path, bbox_inches='tight')
    plt.close(fig)
    
    # Bad channel detection 1, looser, deletion and save-remove severe artifacts
    bads = _detect_bad_channels(raw_filtered, paths, min_peak=10e-6, max_peak=100e-6, pre_ica=True)
    raw_filtered.info['bads'] = bads
    raw_bads_dropped = raw_filtered.drop_channels(bads)
    raw_bads_dropped.save(paths['bad_channels_1_fif'], overwrite=True)
    fig = raw_bads_dropped.compute_psd().plot(show=False) 
    png_path = str(paths['bad_channels_1_fif']).replace('.fif', '_psd.png')
    fig.savefig(png_path, bbox_inches='tight')
    plt.close(fig)
    
    # ICA and save
    raw_ica = _ica_cleaning(raw_bads_dropped, paths)
    raw_ica.save(paths['ica_fif'], overwrite=True)
    fig = raw_ica.compute_psd().plot(show=False)
    png_path = str(paths['ica_fif']).replace('.fif', '_psd.png')
    fig.savefig(png_path, bbox_inches='tight')
    plt.close(fig)

    # Bad channel detection 2-stricter, deletion and save -remove residual artifacts
    bads = _detect_bad_channels(raw_ica, paths, min_peak=10e-6, max_peak=100-6, pre_ica=False)
    raw_ica.info['bads'] = bads
    raw_bads_dropped_2 = raw_ica.drop_channels(bads)
    raw_bads_dropped_2.save(paths['bad_channels_2_fif'], overwrite=True)
    fig = raw_bads_dropped_2.compute_psd().plot(show=False)
    png_path = str(paths['bad_channels_2_fif']).replace('.fif', '_psd.png')
    fig.savefig(png_path, bbox_inches='tight')
    plt.close(fig)
    
    return raw_bads_dropped_2

def flag_eeg_artifacts_from_motion(raw, motion_df):
    """Annotate EEG with motion artifacts, handling sampling rate disparities."""
    # Synchronize motion data to EEG timestamps via interpolation
    artifact_mask = motion_df['MotionArtifact'].values.astype(float)
    timestamps_motion = motion_df['Timestamp'].values
    
    # Interpolate to EEG timeline
    motion_artifact_interp = interpolate.interp1d(timestamps_motion, artifact_mask, kind='previous', bounds_error=False, fill_value=0.0
    )(raw.times)
    artifact_mask_synced = (motion_artifact_interp > 0.5).astype(bool)

    # Find contiguous blocks
    diff_mask = np.diff(artifact_mask_synced.astype(int), prepend=0, append=0)
    onsets_indices = np.where(diff_mask == 1)[0]
    ends_indices = np.where(diff_mask == -1)[0] - 1

    onsets = raw.times[onsets_indices]
    ends = raw.times[ends_indices]
    durations = ends - onsets

    # Handle empty case
    if len(onsets) == 0:
        return raw

    # Create annotations
    annotations = mne.Annotations(onset=onsets, duration=durations, description='bad_motion_artifact')
    raw.set_annotations(raw.annotations + annotations)
    return raw

def _create_raw(stream):
    eeg_data = np.array(stream["time_series"]).T / 1e6  # μV→V
    sfreq = float(stream["info"]["nominal_srate"][0])
    ch_names = "AF3,AF4,F3,Fz,F4,P3,Pz,P4".split(',')[:eeg_data.shape[0]]
    
    # Create info with standard electrode positions
    info = mne.create_info(ch_names, sfreq, 'eeg')
    info.set_montage('standard_1005', match_case=False)
    
    return mne.io.RawArray(eeg_data, info)

def _basic_filtering(raw):
    # Bandpass filter for project compatibility (preserve theta, alpha, beta)
    raw.filter(2, 31, fir_design='firwin')
    
    # Notch filter
    raw.notch_filter(50)
    return raw

def _detect_bad_channels(raw, paths, min_peak, max_peak, pre_ica):
    """Bad channel detection"""       
    # Calculate PTP values
    try:
        ptp_values = _calculate_ptp_percentile(raw, percentile=95)
    except ValueError as e:
        raise RuntimeError(f"PTP calculation failed: {str(e)}")

    # Identify bad channels using ptp_values
    bad_by_ptp = [
        raw.ch_names[i] 
        for i, ptp in enumerate(ptp_values)
        if (ptp < min_peak) or (ptp > max_peak)
    ]
    
    # Final validation
    good_channels = [ch for ch in raw.ch_names if ch not in bad_by_ptp]
    has_frontal = any(ch.startswith('F') for ch in good_channels)
    has_parietal = any(ch.startswith('P') for ch in good_channels)
    
    # Prepare detailed results
    details = {
        "bad_by_ptp": bad_by_ptp,
        "ptp_values": {ch: f"{ptp:.2e} V" for ch, ptp in zip(raw.ch_names, ptp_values)},
        "frontal_remaining": has_frontal,
        "parietal_remaining": has_parietal
    }
    
    # Build log content
    log_content = (
        f"{'Pre-ICA Step' if pre_ica else 'Post-ICA Step'}\n"
        "Bad Channel Report\n"
        "---------------------------\n"
        f"- Bad by peak-to-peak amplitude: {details['bad_by_ptp']}\n"
        f"- Frontal channels remaining: {has_frontal}\n"
        f"- Parietal channels remaining: {has_parietal}\n"
        f"- PTP values:\n"
    ) + "\n".join([f"  {ch}: {val}" for ch, val in details['ptp_values'].items()])
    
    # Add critical error if missing regions
    if not (has_frontal and has_parietal):
        log_content += "\n\nERROR: Insufficient channels for index calculation - "
        log_content += "requires at least one frontal and one parietal channel!"
    
    if pre_ica:
        with open(paths['bad_channels_1_log'], 'w') as f:        
            f.write(log_content)
    else:
        with open(paths['bad_channels_2_log'], 'w') as f:        
            f.write(log_content)
    
    return bad_by_ptp

def _ica_cleaning(raw, paths):
    """Perform automated ICA artifact rejection for eye blinks."""
    # Ensure figures directory exists
    os.makedirs(paths['figures'], exist_ok=True)
    
    #Calculate data rank
    rank = mne.compute_rank(raw, rank='info')['eeg']
    
    # Initialize and fit ICA
    ica = ICA(
        n_components=rank,
        method='infomax',
        max_iter=1000,
        random_state=42,
        fit_params=dict(extended=True)
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
        ic_labels = label_components(raw, ica, method='iclabel')
    
        # Automatically exclude eye components
        auto_excluded = [
            idx for idx, label in enumerate(ic_labels['labels'])
            if 'eye' in label.lower()
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

    if ica.exclude:
        components_kept = [c for c in range(ica.n_components_) if c not in ica.exclude]
    else:
        components_kept = list(range(ica.n_components_))

    var_explained = ica.get_explained_variance_ratio(raw, components=components_kept)['eeg']

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

    return ica.apply(raw)


def _calculate_ptp_percentile(raw, percentile):
    """Calculate specified percentile of PTP in sliding windows"""
    # Create events with duration validation
    duration = 1.0
    if raw.times[-1] < duration:
        raise ValueError(f"Data too short ({raw.times[-1]:.1f}s) for {duration}s windows")
    
    events = make_fixed_length_events(raw, duration=duration, overlap=0.5)
    
    # Create epochs
    epochs = mne.Epochs(
        raw,
        events,
        tmin=0,
        tmax=duration,
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