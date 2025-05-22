import os
from main_batch import main


def process_directory(input_dir):
    """Process all XDF files in a directory"""
    xdf_files = [
        os.path.join(input_dir, f) 
        for f in os.listdir(input_dir) 
        if f.endswith('.xdf') and os.path.isfile(os.path.join(input_dir, f))
    ]
    
    for file_path in xdf_files:
        print(f"\n{'='*40}\nProcessing: {os.path.basename(file_path)}\n{'='*40}")
        try:
            main(file_path)
        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}")
            continue