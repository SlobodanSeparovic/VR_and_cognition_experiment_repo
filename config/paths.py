from pathlib import Path
import os
import re

def get_project_root(marker="settings.ini"):
    """Find project root by searching for a marker file."""
    path = Path(__file__).resolve()
    for parent in [path] + list(path.parents):
        if (parent / marker).exists():
            return parent
    raise FileNotFoundError(f"Could not find project root with marker '{marker}'.")

def get_paths(input_file, base_output_dir=None):
    """Generate paths without session numbers in the directory structure."""
    # Extract subject and task (ignore session)
    base_name = os.path.basename(input_file)
       
    # Subject extraction
    subject = re.search(r'sub-([A-Za-z0-9]+)', base_name).group(1)
    
    # Task extraction: match everything after 'task-' up to the next underscore
    task = re.search(r'task-([a-zA-Z0-9_]+?)(?=_run|$)', base_name).group(1)

    # Set base directory (auto-create /data if not specified)
    if not base_output_dir:
        base_output_dir = get_project_root() / 'data'

    # Define directory structure WITHOUT sessions
    path_template = {
        'root': Path(base_output_dir) / subject / task,
        'raw': '00_raw',
        'filtered': '01_filtered',
        'bad_channels': '02_bad_channels',
        'ica': '03_ica_cleaned',
        'figures': 'reports/figures',
        'logs': 'reports/logs'
    }

    # Create directories and paths
    paths = {}
    for key, rel_path in path_template.items():
        full_path = rel_path if key == 'root' else path_template['root'] / rel_path
        full_path.mkdir(parents=True, exist_ok=True)
        paths[key] = str(full_path)

    # Add file-specific paths
    base_stem = base_name.replace('.xdf', '')
    paths.update({
        'subject': subject,
        'task': task,
        'raw_fif': Path(paths['raw']) / f'{base_stem}_raw.fif',
        'filtered_fif': Path(paths['filtered']) / f'{base_stem}_filtered.fif',
        'bad_channels_fif': Path(paths['bad_channels']) / f'{base_stem}_badchansremoved.fif',
        'ica_fif': Path(paths['ica']) / f'{base_stem}_ica_cleaned.fif',
        'processing_log': Path(paths['logs']) / f'{base_stem}_processing_log.txt',
        'ica_report': Path(paths['figures']) / f'{base_stem}_ica_report.png',
        'ica_rejection_log': Path(paths['logs']) / f'{base_stem}_ica_rejection.txt' 
    })

    return paths
