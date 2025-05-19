import mne
from mne.preprocessing import ICA
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
from pyprep.find_noisy_channels import NoisyChannels
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
    
    # Basic processing (filtering) and save
    raw_filtered = _basic_filtering(raw, paths)
    raw_filtered.save(paths['filtered_fif'], overwrite=True)
    
    # Bad channel detection and deletion and save
    bads = _detect_bad_channels(raw_filtered, paths)
    raw_filtered.info['bads'] = bads
    raw_bads_dropped = raw_filtered.drop_channels(raw.info['bads'])
    raw_bads_dropped.save(paths['bad_channels_fif'], overwrite=True)
    
    # ICA and save
    raw_clean = _ica_cleaning(raw_bads_dropped, paths)
    raw_clean.save(paths['ica_fif'], overwrite=True)
    
    
    return raw_clean

def flag_eeg_artifacts_from_motion(raw, motion_df):
    """
    Annotate EEG with motion artifacts based on pre-buffered motion_df.
    Each contiguous block of MotionArtifact==True is marked as one annotation.
    """
    # Find contiguous artifact blocks
    artifact_mask = motion_df['MotionArtifact'].values
    timestamps = motion_df['Timestamp'].values
    sfreq = 1 / np.mean(np.diff(timestamps))  # Calculate sampling rate

    onsets = []
    durations = []

    in_artifact = False
    for i in range(len(artifact_mask)):
        if artifact_mask[i] and not in_artifact:
            # Start of new artifact block
            start_time = timestamps[i]
            in_artifact = True
        elif not artifact_mask[i] and in_artifact:
            # End of artifact block
            end_time = timestamps[i-1]
            onsets.append(start_time)
            durations.append(end_time - start_time + (1/sfreq))  # Add frame duration
            in_artifact = False
    # Handle case where last sample is in artifact
    if in_artifact:
        end_time = timestamps[-1]
        onsets.append(start_time)
        durations.append(end_time - start_time + (1/sfreq))

    # Create and add annotations
    annotations = mne.Annotations(onset=onsets, duration=durations, description='bad_motion_artifact')
    raw.set_annotations(annotations + raw.annotations)
    return raw

def _create_raw(stream):
    eeg_data = np.array(stream["time_series"]).T / 1e6  # μV→V
    sfreq = float(stream["info"]["nominal_srate"][0])
    ch_names = "AF3,AF4,F3,Fz,F4,P3,Pz,P4".split(',')[:eeg_data.shape[0]]
    
    # Create info with standard electrode positions
    info = mne.create_info(ch_names, sfreq, 'eeg')
    info.set_montage('standard_1020', match_case=False)
    
    return mne.io.RawArray(eeg_data, info)

def _basic_filtering(raw, paths):
    # Filtering
    raw.filter(1, 40, fir_design='firwin')
    raw.notch_filter(50)
    
    # Save PSD plot as PNG
    psd = raw.compute_psd(fmax=40)
    fig_psd = psd.plot(show=False)
    psd_path = os.path.join(paths['figures'], 'psd_before_cleaning.png')
    fig_psd.savefig(psd_path, bbox_inches='tight')
    plt.close(fig_psd)

    return raw

def _detect_bad_channels(raw, paths):
    """Bad channel detection optimized for Enobio 8 dry electrodes"""
    # Ensure montage is set
    if not raw.info['dig']:
        raw.set_montage('standard_1020', match_case=False, on_missing='warn')
    
    # Initialize detector
    noisy_detector = NoisyChannels(
        raw,
        random_state=42,
        matlab_strict=False
    )
    
    # 1. Peak-to-Peak thresholds
    ptp_thresholds = {
        'min_peak': 50e-6,   
        'max_peak': 5000e-6   # 2500 µV
    }
    
    # 2. Calculate PTP values
    try:
        ptp_values = _calculate_ptp_percentile(raw, percentile=95)
    except ValueError as e:
        raise RuntimeError(f"PTP calculation failed: {str(e)}")

    # 3. Identify bad channels using ptp_values
    bad_by_ptp = [
        raw.ch_names[i] 
        for i, ptp in enumerate(ptp_values)
        if (ptp < ptp_thresholds['min_peak']) or (ptp > ptp_thresholds['max_peak'])
    ]
    
    # 4. Adjusted correlation detection
    noisy_detector.find_bad_by_correlation(
        correlation_secs=2,        # 2s windows
        correlation_threshold=0.25, # 0.25 correlation cutoff
        frac_bad=0.25              # 25% bad windows allowed
    )
    
    # 5. Adjusted deviation detection
    noisy_detector.find_bad_by_deviation(
        deviation_threshold= 3.0    # 3.0 SD from median
    )
    
    # Combine all bad channels
    all_bads = list(set(
        bad_by_ptp +
        noisy_detector.bad_by_correlation +
        noisy_detector.bad_by_deviation
    ))
    
    # Final validation
    if len(all_bads) == len(raw.ch_names):
        raise ValueError("All channels flagged as bad! Check preprocessing.")
    if len(all_bads) > len(raw.ch_names) // 2:
        print("Over 50% channels flagged - manual inspection recommended")
    
    # Prepare detailed results
    details = {
        "bad_by_ptp": bad_by_ptp,
        "bad_by_correlation": noisy_detector.bad_by_correlation,
        "bad_by_deviation": noisy_detector.bad_by_deviation,
        "ptp_values": {ch: f"{ptp:.2e} V" for ch, ptp in zip(raw.ch_names, ptp_values)}
    }
    
    # Save bad channel detection report
    log_content = (
        "Bad Channel Report\n"
        "---------------------------\n"
        f"- Bad by peak-to-peak amplitude: {details['bad_by_ptp']}\n"
        f"- Bad by correlation: {details['bad_by_correlation']}\n"
        f"- Bad by deviation: {details['bad_by_deviation']}\n"
        f"- Final bad channels: {all_bads}\n"
    )
    with open(paths['processing_log'], 'w') as f:        
        f.write(log_content)
    
    return all_bads

def _ica_cleaning(raw, paths):
    """Perform hybrid automated/manual ICA artifact rejection."""
    # Ensure figures directory exists
    os.makedirs(paths['figures'], exist_ok=True)
    
    # Initialize and fit ICA
    ica = ICA(
        n_components=0.95,        # Retains 95% variance (good for noisy dry EEG)
        method='infomax',         # Handles sub-Gaussian sources (e.g., blinks)
        max_iter=1000,            # ICA converges better in high-noise scenarios
        random_state=42,          # Ensures reproducibility
        fit_params=dict(extended=True)  # Better for non-Gaussian artifacts
    )
    ica.fit(raw)

    # Create component plots directory
    components_dir = os.path.join(paths['figures'], 'ica_components')
    os.makedirs(components_dir, exist_ok=True)

    # ====================
    # Automated Exclusion Phase
    # ====================
    auto_excluded = []
    try:
        ic_labels = label_components(raw, ica, method="iclabel")
        
        # Automatically exclude eye components with >65% confidence
        auto_excluded = [
            idx for idx, (label, prob) in enumerate(zip(
                ic_labels['labels'], 
                ic_labels['y_pred_proba']
            )) 
            if label == 'eye' and prob > 0.65
        ]
        
    except Exception as e:
        print(f"ICLabel Error: {str(e)} - Proceeding to manual inspection")
    
    ica.exclude = auto_excluded

    # Manual Fallback Phase
    # ====================
    if not ica.exclude:
        print("\nNo components met automatic exclusion criteria.")
        print("Launching manual component inspection...")

        # 1. Generate component plots FIRST to avoid output spam
        print("\nGenerating component detail plots...")
        for comp in range(ica.n_components_):
            fig_comp = ica.plot_properties(raw, picks=comp, show=False)
            fig_comp[0].savefig(os.path.join(components_dir, f'component_{comp:02d}.png'),
                                bbox_inches='tight')
            plt.close(fig_comp[0])
        print("Component details saved to:", components_dir)

        # 2. Create optimized figure layout
        fig = plt.figure(figsize=(12, 7), dpi=100, num="ICA Component Inspection")  # Reduced size
        gs = gridspec.GridSpec(3, 1, height_ratios=[1.5, 1.5, 1], hspace=0.4)  # Simplified grid

        # Component grid (2x4)
        comp_grid = gridspec.GridSpecFromSubplotSpec(2, 4, subplot_spec=gs[0:2], hspace=0.2, wspace=0.15)
        axes = []
        for i in range(min(ica.n_components_, 8)):
            ax = fig.add_subplot(comp_grid[i//4, i%4])
            ica.plot_components(picks=i, axes=ax, show=False, res=32)  # Lower resolution
            ax.set_title(f"Component {i}", fontsize=9)  # Smaller title

        # Text panel with manual spacing
        ax_text = fig.add_subplot(gs[2])
        ax_text.axis('off')

        # Compact text formatting
        class_text = ["ICLabel classifications (Close window to exclude components):"]
        if 'ic_labels' in locals():
            for comp in range(ica.n_components_):
                label = ic_labels['labels'][comp]
                prob = ic_labels['y_pred_proba'][comp]
                class_text.append(f"ICA Component {comp}: {prob:.0%} confident this is {label}{'*' if label == 'other' else ''}")
            class_text.append("")
            class_text.append("Confidence percentages indicate the classifier's confidence in the label.")
            class_text.append("*ICs containing indeterminate noise or ICs containing multiple signals that ICA decomposition could not separate well.")
        else:
            class_text.append("ICLabel unavailable")

        ax_text.text(0.01, 0.95, "\n".join(class_text),
                    fontfamily='monospace', fontsize=8,  # Smaller font
                    verticalalignment='top',
                    transform=ax_text.transAxes)

        # Manual layout adjustments (avoids tight_layout warning)
        plt.subplots_adjust(left=0.05, right=0.95, top=0.92, bottom=0.08)
        plt.show(block=True)

        # 3. Get user input after closing figure
        ica.exclude = _get_valid_exclusion_input(ica.n_components_)
        plt.close('all')

    # Reporting Phase
    # ====================

    fig = plt.figure(figsize=(14, 8), dpi=100)
    gs = gridspec.GridSpec(nrows=3, ncols=4, height_ratios=[2, 2, 1])

    # Plot all components in a 2x4 grid (up to 8 components)
    for i in range(min(ica.n_components_, 8)):
        ax = fig.add_subplot(gs[i // 4, i % 4])
        ica.plot_components(picks=i, axes=ax, show=False, res=32)
        # Mark excluded with a big red X
        if i in ica.exclude:
            ax.text(0.5, 0.5, 'X', color='red', fontsize=50, ha='center', va='center', transform=ax.transAxes, weight='bold')
        ax.set_title(f"ICA Component {i}", fontsize=9)

    # ICLabel results and instructions in a wide text box at the bottom
    ax_text = fig.add_subplot(gs[2, :])
    ax_text.axis('off')

    var_explained = ica.get_explained_variance_ratio(
        raw, 
        components=[c for c in range(ica.n_components_) if c not in ica.exclude]
    )['eeg']

    report_text = [
        "ICA ARTIFACT REJECTION REPORT",
        "============================",
        f"Exclusion mode: {'AUTOMATIC' if auto_excluded else 'MANUAL'}",
        f"Excluded components: {ica.exclude}",
        f"Variance retained: {var_explained:.1%}",
        "",
        "ICLabel classifications for each component:"
    ]

    if 'ic_labels' in locals() and ic_labels is not None:
        for comp in range(ica.n_components_):
            label = ic_labels['labels'][comp]
            prob = ic_labels['y_pred_proba'][comp]
            line = f"ICA Component {comp}: {prob:.0%} confident this is {label}"
            if label == 'other':
                line += "*"
            report_text.append(line)
    else:
        report_text.append("ICLabel classification unavailable.")

    report_text.append("")
    report_text.append("Confidence percentages indicate the classifier's confidence in the label.")
    report_text.append("*ICs containing indeterminate noise or ICs containing multiple signals that ICA decomposition could not separate well.")

    ax_text.text(
        0.01, 0.99, "\n".join(report_text),
        fontfamily='monospace',
        fontsize=8,
        linespacing=1.2,
        verticalalignment='top',
        transform=ax_text.transAxes,
        bbox=dict(facecolor='white', alpha=0.95)
    )

    plt.tight_layout(pad=1.0)
    plt.subplots_adjust(top=0.92, bottom=0.08)

    # Save and close report
    save_path = os.path.join(paths['figures'], 'ica_report.png')
    fig.savefig(save_path, bbox_inches='tight')
    plt.close(fig)

    # Save detailed log
    with open(paths['ica_rejection_log'], 'w') as f:
        f.write("\n".join(report_text))
        if ica.exclude:
            f.write("\nExclusion rationale:\n")
            if auto_excluded:
                f.write("- Automatic: Eye components with >80% confidence\n")
            else:
                f.write("- Manual: User-selected during inspection\n")

    return ica.apply(raw)


def _get_valid_exclusion_input(n_components):
    """Validate component exclusion input with enhanced checks."""
    while True:
        exclude = input("Enter components to exclude (comma-separated): ").strip()
        if not exclude:
            return []
        try:
            excluded = list(map(int, exclude.split(',')))
            unique_excluded = list(set(excluded))  # Remove duplicates
            if all(0 <= c < n_components for c in unique_excluded):
                return unique_excluded
            print(f"Invalid components! Valid range: 0-{n_components-1}")
        except ValueError:
            print("Invalid format. Use numbers like '0,2,3'")


def _calculate_ptp_percentile(raw, percentile=95):
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