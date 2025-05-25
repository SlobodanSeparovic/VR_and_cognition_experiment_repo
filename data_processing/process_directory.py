import os
import re
from pathlib import Path
from main_single import main

def process_directory(root_dir):
    """Process all XDF files in BIDS-style directory"""
    # Collect all XDF files
    xdf_paths = set()
    
    for dirpath, _, filenames in os.walk(root_dir):
        if re.search(r'sub-[^/\\]+[/\\]ses-S001[/\\]eeg$', dirpath.replace('\\', '/')):
            for f in filenames:
                if f.endswith('.xdf'):
                    full_path = os.path.abspath(os.path.join(dirpath, f))
                    print(f"Found: {full_path}")  # Debug print
                    xdf_paths.add(full_path)
    
    # Process collected files
    for file_path in xdf_paths:
        try:
            main(file_path)
        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}")
