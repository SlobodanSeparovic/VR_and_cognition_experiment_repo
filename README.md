# VR and cognition experiment data processing pipeline

# Structure
```
VR_and_cognition_experiment/
├── config/
│   └── paths.py
├── data_processing/
│   ├── analyze_eeg.py
│   ├── preprocess_behavioral.py
│   ├── preprocess_eeg.py
│   ├── preprocess_vr_motion.py
│   ├── process_directory.py
│   ├── read_xdf.py
│   └── update_participant_data.py
├── experiment_scripts/
│   ├── nasatlx_stream.py
│   ├── random_seqs.csv
│   ├── randomization.py
├── .gitignore
├── main_analysis.py
├── main_correction.py
├── main_preprocessing.py
├── README.md
├── requirements.txt
└── settings.ini
```

# Background
This is the data processing pipeline for the data recorded using the experimental setup stated below. The core idea of the experiment was to explore if and how different environments (control, nature and urban) may pose different level of cognitive strain on participants. The results are meant to inform the chioce of different environments in post-stroke rehabilitation using virtual-reality headsets.

# Local data handling
The pipeline extracts root repository directory used locally for cloning. It creates paths for newly created files. If they are already available locally, it overwrites the old files. For the required structure of the paths for the participants' data to be processed, please refer to the brAIn lab data repository used to store participant data from this experiment. The files from participants need to be stored in experiment folder, and follow the same naming convention that is BIDS-compliant. 

# Experimental setup using Unity
The experimental setup with different environments can be found in the github repository https://github.com/carolinarve/setup_repo

# Data collected during the experiments
The database of experimental data can be found in the brAIn lab github repository https://github.itu.dk/brAInLab/vr-experiment-data.git

# Auxiliary scripts and data
In the folder experiment_scripts you may find auxiliarry scripts and one cvs file. The nasatlx_stream.py was used for collection of subjective NASAtlx cognitive load index. Randomization.py was developed to generate randomization sequence of baseline measurements aand experimental conditions. Random_seqs.csv is a file containing the order of baseline and experimental conditions for 60 participants created by randomization.py that was followed in our experiment.

# Process
1. Open settings.ini

2. Add .xdf file path to settings.ini without quotation marks. Using this path, thy pipeline creates local folders and stores data.

3. Add preferred mode of processing . Use 1 for processing the exact file in the specified path. Use 2 to process all XDF files that are located in the same participant's directory as the file specified in the path. Use 3 if you want to process all participants' data that are strored in the experiment directory.

4. open and run main.py

5. This results with files stored under /data that contain logs and informations on results from individual steps of eeg data processing, and with a CVS file stored under /results wihtin respective participant's folder containing raw data for further processing.

4. Make sure data from all participants have gone through this stage.

5. After all participants have been processed open and run main_data_correction.

6. This generates CVS files that contain corrected eeg-derived indexes, i.e. ther change relative to the respective baseline. The formula used is (condition-baseline)/baseline. These files are stored in the same folder as CVS file contining respective participant's raw data. Two more CVS files are generated in the /results folder. The first is named final_dataset.cvs and it contains encoded variables from all participants suited for statistical analysis. Nominal categorical data are one-hot-encoded. Continuos data are left as is. The second CVS file is called motion_artifacts.csv and reflects the proportion of data in eeg that was flagged as motion artifact and was excluded from power spectral densities' calculations. This file only serves for reporting purposes.
