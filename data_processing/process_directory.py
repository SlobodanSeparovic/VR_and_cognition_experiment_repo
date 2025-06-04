import os
import re
from pathlib import Path
from main_preprocessing import main_single

def process_directory(directory):
    """Process all XDF files in the directory"""
    # Collect all XDF files
    xdf_paths = set()
    
    for dirpath, _, filenames in os.walk(directory):
        if re.search(r'sub-[^/\\]+[/\\]ses-S001[/\\]eeg$', dirpath.replace('\\', '/')):
            for f in filenames:
                if f.endswith('.xdf'):
                    full_path = os.path.abspath(os.path.join(dirpath, f))
                    xdf_paths.add(full_path)
    
    # Convert to sorted list in alphabetical order
    sorted_xdf_paths = sorted(xdf_paths)
    
    # Process collected files
    for file_path in sorted_xdf_paths:
        try:
            main_single(file_path)
        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}")
