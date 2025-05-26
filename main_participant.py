import configparser
import os
from data_processing.process_directory import process_directory

# Read settings.ini from the repo root
config = configparser.ConfigParser()
# Read settings.ini from the same directory as main.py
config_path = os.path.join(os.path.dirname(__file__), "settings.ini")
config.read(config_path)
input_file = config['INPUT']['input_file']

if __name__ == "__main__":
    # Get the directory threetwo levels up from the file (which is 'experiment')
    bids_root = os.path.abspath(os.path.join(os.path.dirname(input_file), '../..'))
    process_directory(bids_root)