import mne
from mne.preprocessing import ICA
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
from pyprep.find_noisy_channels import NoisyChannels
from mne_icalabel import label_components
from mne import make_fixed_length_events

def process_eeg(stream, paths, start_time, end_time):
    raw = create_raw(stream)
    raw_processed = basic_processing(raw, paths)
    
    # Convert absolute times to relative
    eeg_start_time = stream["time_stamps"][0]
    tmin = max(0, start_time - eeg_start_time)  # Ensure non-negative
    tmax = min(raw.times[-1], end_time - eeg_start_time)  # Don't exceed EEG duration
    
    raw_processed.crop(tmin=tmin, tmax=tmax)
    raw_clean = ica_cleaning(raw_processed, paths)
    
    # Save using context manager
    with mne.io.RawArray(raw_clean.get_data(), raw_clean.info) as temp_raw:
        temp_raw.save(os.path.join(paths['processed'], 'cleaned_eeg.fif'), overwrite=True)
    
    return raw_clean

def create_raw(stream):
    eeg_data = np.array(stream["time_series"]).T / 1e6  # μV→V
    sfreq = float(stream["info"]["nominal_srate"][0])
    ch_names = "AF3,AF4,F3,Fz,F4,P3,Pz,P4".split(',')[:eeg_data.shape[0]]
    
    # Create info with standard electrode positions
    info = mne.create_info(ch_names, sfreq, 'eeg')
    info.set_montage('standard_1020', match_case=False)
    
    return mne.io.RawArray(eeg_data, info)

def basic_processing(raw, paths):
    # Filtering
    raw.filter(1, 40, fir_design='firwin')
    raw.notch_filter(50)
    
    # Bad channel detection and interpolation
    bads, bad_details = detect_bad_channels(raw)
    raw.info['bads'] = bads
    raw.drop_channels(raw.info['bads'])

    # Save PSD plot as PNG
    psd = raw.compute_psd(fmax=40)
    fig_psd = psd.plot(show=False)
    psd_path = os.path.join(paths['figures'], 'psd_before_cleaning.png')
    fig_psd.savefig(psd_path, bbox_inches='tight')
    plt.close(fig_psd)

    # Save bad channel detection report
    log_content = (
        "Bad Channel Report\n"
        "---------------------------\n"
        f"- Bad by peak-to-peak amplitude: {bad_details['bad_by_ptp']}\n"
        f"- Bad by correlation: {bad_details['bad_by_correlation']}\n"
        f"- Bad by deviation: {bad_details['bad_by_deviation']}\n"
        f"- Final bad channels: {bads}\n"
    )
    with open(paths['processing_log'], 'w') as f:
        f.write(log_content)

    return raw


def ica_cleaning(raw, paths):
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

    # ====================
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
        ica.exclude = get_valid_exclusion_input(ica.n_components_)
        plt.close('all')

    # ====================
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


def safe_topo_plot(ica, temp_path, exclude=None):
    """Safe topography plotting with excluded components marked."""
    os.makedirs(os.path.dirname(temp_path), exist_ok=True)
    
    try:
        # Plot all components
        topo_fig = ica.plot_components(show=False)
        
        # Mark excluded components with red X
        if exclude:
            fig = topo_fig[0] if isinstance(topo_fig, list) else topo_fig
            for idx in exclude:
                if idx < len(fig.axes):
                    ax = fig.axes[idx]
                    ax.text(0.5, 0.5, 'X', color='red', fontsize=48,
                            ha='center', va='center', transform=ax.transAxes,
                            weight='bold')

        # Save and close
        if isinstance(topo_fig, list):
            topo_fig[0].savefig(temp_path, bbox_inches='tight')
            [plt.close(f) for f in topo_fig]
        else:
            topo_fig.savefig(temp_path, bbox_inches='tight')
            plt.close(topo_fig)
            
    except Exception as e:
        raise RuntimeError(f"Failed to create topography plot: {str(e)}")

def generate_iclabel_text(raw, ica):
    """Generate ICLabel classification text with error handling."""
    try:
        ic_labels = label_components(raw, ica, method="iclabel")
        return "\n".join(
            f"Component {c}: {ic_labels['labels'][c]} ({ic_labels['y_pred_proba'][c]:.1%})"
            for c in range(ica.n_components_)
        )
    except Exception as e:
        return f"ICLabel Error: {str(e)}"

def get_valid_exclusion_input(n_components):
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


def detect_bad_channels(raw):
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
        'min_peak': 150e-6,   # 150 µV
        'max_peak': 2500e-6   # 2500 µV
    }
    
    # 2. Calculate PTP values
    try:
        ptp_values = calculate_ptp_percentile(raw, percentile=95)
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
        correlation_secs=4,        # 4s windows
        correlation_threshold=0.3, # 0.3 correlation cutoff
        frac_bad=0.15              # 15% bad windows allowed
    )
    
    # 5. Adjusted deviation detection
    noisy_detector.find_bad_by_deviation(
        deviation_threshold=4.5    # 4.5 SD from median
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
        all_bads = []  # Fallback to empty list
    
    # Prepare detailed results
    details = {
        "bad_by_ptp": bad_by_ptp,
        "bad_by_correlation": noisy_detector.bad_by_correlation,
        "bad_by_deviation": noisy_detector.bad_by_deviation,
        "ptp_values": {ch: f"{ptp:.2e} V" for ch, ptp in zip(raw.ch_names, ptp_values)}
    }
    
    return all_bads, details

def calculate_ptp_percentile(raw, percentile=95):
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