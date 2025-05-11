from pathlib import Path
import os

def get_project_root(marker="settings.ini"):
    """Find project root by searching for a marker file."""
    path = Path(__file__).resolve()
    for parent in [path] + list(path.parents):
        if (parent / marker).exists():
            return parent
    raise FileNotFoundError(f"Could not find project root with marker '{marker}'.")

def get_paths(input_file):
    """
    Generate standardized paths for data and report storage based on input file.
    Returns a dict with all relevant paths for figures and text logs.
    """
    # Extract subject and session from filename
    base_name = os.path.basename(input_file)
    parts = base_name.split('_')
    subject = [p for p in parts if p.startswith('sub-')][0]
    session = [p for p in parts if p.startswith('ses-')][0]

    # Define base directories
    base_dir = get_project_root()
    base_data_dir = os.path.join(base_dir, 'data')
    base_report_dir = os.path.join(base_dir, 'report')

    # Construct all needed directories
    processed_dir = os.path.join(base_data_dir, subject, session, 'eeg')
    reports_dir = os.path.join(base_report_dir, subject, session, 'eeg')
    figures_dir = os.path.join(base_report_dir, subject, session, 'figures')

    # Ensure directories exist
    for d in [processed_dir, reports_dir, figures_dir]:
        os.makedirs(d, exist_ok=True)

    # Example file names for data and logs (no HTML)
    raw_edf = os.path.join(processed_dir, base_name.replace('.xdf', '.edf'))
    cleaned_fif = os.path.join(processed_dir, base_name.replace('.xdf', '_cleaned.fif'))
    processing_log = os.path.join(reports_dir, 'processing_log.txt')
    ica_rejection_log = os.path.join(reports_dir, 'ica_rejection.txt')

    return {
        'subject': subject,
        'session': session,
        'base_data_dir': base_data_dir,
        'base_report_dir': base_report_dir,
        'processed': processed_dir,
        'reports': reports_dir,
        'figures': figures_dir,
        'raw_edf': raw_edf,
        'cleaned_fif': cleaned_fif,
        'processing_log': processing_log,
        'ica_rejection_log': ica_rejection_log,
    }