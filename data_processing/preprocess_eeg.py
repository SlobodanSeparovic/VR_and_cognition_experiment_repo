import mne
from mne.preprocessing import ICA
from mne.report import Report
import numpy as np
import matplotlib.pyplot as plt
import os
from pyprep.find_noisy_channels import NoisyChannels

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
    raw.interpolate_bads()

    # Save PSD plot as PNG
    psd = raw.compute_psd(fmax=40)
    fig_psd = psd.plot(show=False)
    psd_path = os.path.join(paths['figures'], 'psd_before_cleaning.png')
    fig_psd.savefig(psd_path, bbox_inches='tight')
    plt.close(fig_psd)

    # Save RANSAC report
    log_content = (
        "Bad Channel Report (RANSAC)\n"
        "---------------------------\n"
        f"- Bad by RANSAC prediction: {bad_details['bad_by_ransac']}\n"
        f"- Bad by correlation: {bad_details['bad_by_correlation']}\n"
        f"- Bad by deviation: {bad_details['bad_by_deviation']}\n"
        f"- Final bad channels: {bads}\n"
    )
    with open(paths['processing_log'], 'w') as f:
        f.write(log_content)

    return raw



def ica_cleaning(raw, paths):
    # Initialize ICA with automatic component selection
    ica = ICA(
        n_components=0.999999,  # Auto-select based on variance
        method='infomax',
        random_state=42,
        fit_params=dict(extended=True)  # Better for non-Gaussian sources
    )

    # Fit ICA 
    ica.fit(raw)

    # Save ICA components plot with version-agnostic handling
    fig_ica = ica.plot_components(show=False)
    ica_path = os.path.join(paths['figures'], 'ica_components.png')
    
    if hasattr(fig_ica, '__iter__'):  # Handle figure list (older MNE)
        main_fig = fig_ica[0]
    else:  # Single figure (newer MNE)
        main_fig = fig_ica
    
    main_fig.savefig(ica_path, bbox_inches='tight')
    plt.close(main_fig)

    # Interactive component selection
    ica.plot_components()
    plt.show()  # Ensure plot stays open for input
    exclude = input("Enter components to exclude (comma-separated): ").strip()
    ica.exclude = list(map(int, exclude.split(','))) if exclude else []

    # Save detailed rejection info
    rejection_content = (
        "ICA Component Rejection Report\n"
        "------------------------------\n"
        f"Excluded components: {ica.exclude}\n"
        f"Total components: {ica.n_components_}\n"
        f"Variance explained: {ica.n_components_ / len(raw.ch_names):.1%}"
    )
    
    with open(paths['ica_rejection_log'], 'w') as f:
        f.write(rejection_content)

    return ica.apply(raw)

def detect_bad_channels(raw):
    """Robust bad channel detection using RANSAC for small arrays"""
    if not raw.info['dig']:
        raw.set_montage('standard_1020', match_case=False)
    
    # Initialize detector with recommended parameters
    noisy_detector = NoisyChannels(
        raw,
        random_state=42,
        matlab_strict=False
    )
    
    # Run RANSAC with adjusted parameters for 8-channel EEG
    noisy_detector.find_bad_by_ransac(
        n_samples=50,          
        sample_prop=0.5,     # 4 channels per sample (8 * 0.5 = 4)
        corr_thresh=0.6,        # Lower threshold for sparse arrays
        frac_bad=0.3           # Require fewer bad windows to flag
    )
    
    # Find other bad channels (correlation, deviation)
    noisy_detector.find_bad_by_correlation()
    noisy_detector.find_bad_by_deviation()

    # Prepare results
    details = {
        "bad_by_ransac": noisy_detector.bad_by_ransac,
        "bad_by_correlation": noisy_detector.bad_by_correlation,
        "bad_by_deviation": noisy_detector.bad_by_deviation
    }
    
    return noisy_detector.get_bads(), details