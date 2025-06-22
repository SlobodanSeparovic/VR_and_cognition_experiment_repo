import os
import pandas as pd
import numpy as np
from pathlib import Path

def process_all_participants(results_dir='results'):
    """Main processing function with environment-specific outlier handling"""
    # Clean up previous outputs
    _delete_old_csvs(results_dir)    
    Path(results_dir).mkdir(parents=True, exist_ok=True)
    
    final_data = []
    motion_artifacts_data = []
    removal_report = []

    tracking = {
        'eligible': {2: set(), 3: set()},
        'excluded': {2: set(), 3: set()},
        'excluded_both': set(),
        'dv_excluded': []
    }

    # First pass: Track initial eligibility
    for participant_dir in Path(results_dir).iterdir():
        if participant_dir.is_dir():
            raw_id = participant_dir.name
            clean_id = raw_id.replace('P', '')
            input_file = participant_dir / f"{raw_id}_data.csv"
            
            if input_file.exists():
                corrected_df, _ = _process_participant_file(input_file)
                if not corrected_df.empty:
                    for n_arm in [2, 3]:
                        if n_arm in corrected_df['N'].astype(int).values:
                            tracking['eligible'][n_arm].add(int(clean_id))
    
    # Second pass: Process data
    for participant_dir in Path(results_dir).iterdir():
        if participant_dir.is_dir():
            raw_id = participant_dir.name
            clean_id = raw_id.replace('P', '')
            input_file = participant_dir / f"{raw_id}_data.csv"
            
            if input_file.exists():
                corrected_df, motion_df = _process_participant_file(input_file)
                
                if not corrected_df.empty:
                    corrected_df.to_csv(participant_dir / f"{raw_id}_data_corrected.csv", index=False)
                    final_data.append(corrected_df)
                    motion_artifacts_data.append(motion_df)

    if final_data:
        final_df = pd.concat(final_data, ignore_index=True).reset_index(drop=True)
        final_path = Path(results_dir) / 'final_dataset.csv'
        
        column_order = [
            'Subject', 'Env', 'N', 'NasaCLI', 'FTCLI_corrected',
            'FTPACLI_corrected', 'EngI_corrected', 'F1', 'TotalPress',
            'IncorrNo', 'CorrRatio', 'RTCorrAv', 'RTIncorrAv'
        ]
        final_df = final_df.reindex(columns=column_order)

        outlier_participants = set()
        nan_columns = {2: set(), 3: set()}

        # Environment-specific outlier detection
        for n_arm in [2, 3]:
            arm_mask = final_df['N'].astype(int) == n_arm
            arm_data = final_df[arm_mask].copy()
            
            # Group by environment within arm
            for env in arm_data['Env'].unique():
                env_mask = (arm_data['Env'] == env)
                env_data = arm_data[env_mask].copy()
                
                for metric in ['FTCLI_corrected', 'FTPACLI_corrected', 'EngI_corrected']:
                    if metric in env_data.columns and len(env_data) >= 4:  # Need min 4 points for IQR
                        # Calculate IQR bounds
                        Q1 = env_data[metric].quantile(0.25)
                        Q3 = env_data[metric].quantile(0.75)
                        IQR = Q3 - Q1
                        lower_bound = Q1 - 1.5 * IQR
                        upper_bound = Q3 + 1.5 * IQR
                        
                        # Identify outliers
                        outliers = env_data[
                            (env_data[metric] < lower_bound) | 
                            (env_data[metric] > upper_bound)
                        ]
                        
                        # Mark participants for exclusion
                        for _, row in outliers.iterrows():
                            participant_id = row['Subject']
                            outlier_participants.add((participant_id, n_arm))
                            tracking['excluded'][n_arm].add(participant_id)
                            
                            removal_report.append({
                                'Participant': participant_id,
                                'Arm': n_arm,
                                'Environment': env,
                                'Metric': metric,
                                'Value': f"{row[metric]:.2f}",
                                'Reason': 'Outlier'
                            })

            # NaN handling
            arm_nan_cols = arm_data.columns[arm_data.isna().any()].tolist()
            if arm_nan_cols:
                final_df.loc[arm_mask, arm_nan_cols] = np.nan
                nan_columns[n_arm].update(arm_nan_cols)
                
                for col in arm_nan_cols:
                    removal_report.append({
                        'Participant': 'All',
                        'Arm': n_arm,
                        'Environment': 'All',
                        'Metric': col,
                        'Value': 'NaN present',
                        'Reason': 'Column nullified'
                    })

        # Remove participants with any outliers in the arm
        exclusion_mask = final_df[['Subject', 'N']].apply(tuple, axis=1).isin(outlier_participants)
        final_df = final_df[~exclusion_mask]

        # Column cleanup
        common_nan_cols = nan_columns[2].intersection(nan_columns[3])
        if common_nan_cols:
            final_df = final_df.drop(columns=list(common_nan_cols))
            tracking['dv_excluded'] = list(common_nan_cols)
            for col in common_nan_cols:
                removal_report.append({
                    'Participant': 'All',
                    'Arm': 'Both',
                    'Environment': 'All',
                    'Metric': col,
                    'Value': 'NaN in both arms',
                    'Reason': 'Column removed'
                })
                
        tracking['excluded_both'] = set(tracking['excluded'][2]) & set(tracking['excluded'][3])

        # Generate reports
        _generate_cleaning_report(
            removal_report,
            tracking['eligible'],
            tracking['excluded'],
            tracking['excluded_both'],
            tracking['dv_excluded'],
            results_dir
        )
        
        # Save outputs
        _overwrite_final_dataset(final_path, final_df)
        _overwrite_final_dataset(
            Path(results_dir)/'motion_artifacts.csv',
            pd.concat(motion_artifacts_data, ignore_index=True)
        )

def _delete_old_csvs(results_dir='results'):
    """Remove previous output files before processing"""
    output_files = [
        'final_dataset.csv',
        'motion_artifacts.csv',
        'data_cleaning_report.md'
    ]
    for fname in output_files:
        file_path = Path(results_dir) / fname
        if file_path.exists():
            os.remove(file_path)
    
    # Remove participant-level corrected files
    for participant_dir in Path(results_dir).iterdir():
        if participant_dir.is_dir():
            for csv_file in participant_dir.glob('*_data_corrected.csv'):
                os.remove(csv_file)

def _generate_cleaning_report(report_data, eligible, excluded, excluded_both, dv_excluded, output_dir):
    """Generate markdown report with environment-level details"""
    report_path = Path(output_dir) / 'data_cleaning_report.md'
    
    with open(report_path, 'w') as f:
        f.write("# Data Cleaning Report\n\n")
        f.write("## Summary\n\nAutomated data cleaning report with environment-specific outlier detection.\n\n")
        
        # Participant exclusions
        f.write("## Participant Exclusions\n\n")
        for arm in [2, 3]:
            f.write(f"### Arm {arm}\n\n")
            f.write(f"- **Eligible participants:** {len(eligible[arm])}\n")
            f.write(f"- **Excluded participants:** {len(excluded[arm])}\n")
            f.write(f"- **Participants in analysis:** {len(eligible[arm]) - len(excluded[arm])}\n")
            excl_list = sorted(list(excluded[arm]))
            f.write(f"- **Exclusion list:** {', '.join(map(str, excl_list))}\n\n")
        
        # Cross-arm exclusions
        f.write("### Cross-Arm Exclusions\n\n")
        f.write(f"- **Excluded from both arms:** {len(excluded_both)}\n")
        cross_excl = sorted(list(excluded_both))
        f.write(f"- **Participants:** {', '.join(map(str, cross_excl))}\n\n")
        
        # Data modifications
        f.write("## Data Modifications\n\n")
        f.write(f"- **Removed variables:** {', '.join(dv_excluded)}\n")
        f.write(f"- **Total modifications:** {len(report_data)} entries\n\n")
        
        # Detailed log
        f.write("## Detailed Modification Log\n\n")
        f.write("| Participant | Arm  | Environment | Metric            | Value             | Reason           |\n")
        f.write("|-------------|------|-------------|-------------------|-------------------|------------------|\n")
        for entry in report_data:
            f.write(
                f"| {str(entry['Participant']).ljust(11)} | "
                f"{str(entry['Arm']).ljust(4)} | "
                f"{str(entry['Environment']).ljust(11)} | "
                f"{entry['Metric'].ljust(17)} | "
                f"{entry['Value'].ljust(17)} | "
                f"{entry['Reason'].ljust(16)} |\n"
            )

def _process_participant_file(file_path):
    """Process participant data with baseline correction"""
    df = pd.read_csv(file_path)
    df['Subject'] = df['Subject'].str.replace('P', '').astype(int)
    df['N'] = df['N'].astype(int)
    
    # Split baseline and task data
    baseline_df = df[df['N'] == 0].copy()
    task_df = df[df['N'] != 0].copy()
    
    if baseline_df.empty:
        return pd.DataFrame(), pd.DataFrame()
    
    # Apply baseline correction
    for metric in ['FTCLI', 'FTPACLI', 'EngI']:
        task_df[f'{metric}_corrected'] = task_df.apply(
            lambda row: _calculate_corrected_value(row, baseline_df, metric),
            axis=1
        )

    output_cols = [
        'Subject', 'Env', 'N', 'NasaCLI', 'FTCLI_corrected',
        'FTPACLI_corrected', 'EngI_corrected', 'F1', 'TotalPress',
        'IncorrNo', 'CorrRatio', 'RTCorrAv', 'RTIncorrAv'
    ]
    
    return (
        task_df[output_cols].sort_values(['Env', 'N']).reset_index(drop=True),
        df[['Subject', 'N', 'Env', 'MotionArt']].copy()
    )

def _calculate_corrected_value(row, baseline_df, column):
    """Calculate baseline-normalized values"""
    try:
        env = row['Env']
        subject = row['Subject']
        
        baseline = baseline_df[
            (baseline_df['Env'] == env) &
            (baseline_df['Subject'] == subject)
        ][column].values[0]
        
        return (row[column] - baseline) / baseline if baseline != 0 else np.nan
    except (IndexError, KeyError):
        return np.nan

def _overwrite_final_dataset(final_path, new_data):
    """Save final cleaned dataset"""
    new_data.to_csv(final_path, index=False)

if __name__ == "__main__":
    process_all_participants()
