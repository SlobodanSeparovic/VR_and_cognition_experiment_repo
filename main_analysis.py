import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import sys
import pingouin as pg
from pathlib import Path
from sklearn.preprocessing import PowerTransformer

# Configure numerical stability
np.seterr(divide='ignore', invalid='ignore')
warnings.filterwarnings("ignore", category=RuntimeWarning, module="numpy")

# Setup base directories and paths
BASE_DIR = Path(__file__).parent
DATA_PATH = BASE_DIR / 'results' / 'final_dataset.csv'
RESULTS_DIR = BASE_DIR / 'results' / 'statistical_analysis'
LOG_FILE = RESULTS_DIR / 'analysis.log'

# Create results directory if needed
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Configure logging system
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE, mode='w'),
        logging.StreamHandler(sys.stdout)
    ]
)

def initialize_directories():
    """Create required output directories"""
    for task in ['2back', '3back']:
        (RESULTS_DIR / task / 'parametric').mkdir(parents=True, exist_ok=True)
        (RESULTS_DIR / task / 'nonparametric').mkdir(parents=True, exist_ok=True)

def load_dataset():
    """Load and validate input dataset from CSV"""
    try:
        df = pd.read_csv(DATA_PATH)
        logging.info(f"Dataset loaded from {DATA_PATH}")
        return df
    except FileNotFoundError:
        logging.critical("Missing input data file - aborting analysis")
        sys.exit(1)

def filter_dvs_by_missing(data, dvs, task_level):
    """Exclude variables with missing values"""
    valid = []
    for dv in dvs:
        if data[dv].isna().any():
            missing_count = data[dv].isna().sum()
            total = len(data[dv])
            logging.warning(f"Excluding {dv} from {task_level} - {missing_count}/{total} ({missing_count/total:.1%}) missing values")
        else:
            valid.append(dv)
    return valid

def check_rm_assumptions(data, dv):
    """Check assumptions for parametric analysis"""
    norm = pg.normality(data[dv])
    sph = pg.sphericity(data, dv=dv, subject='Subject', within='Env')
    
    return {
        'normality_passed': norm['pval'].values[0] >= 0.05,
        'normality_pval': norm['pval'].values[0],
        'sphericity_pval': sph[-1],
        'epsilon_gg': pg.epsilon(data, dv=dv, correction='gg'),
        'valid': (norm['pval'].values[0] >= 0.05) and (sph[-1] >= 0.05)
    }

def generate_report(assumptions, transforms, dv, task_level, save_dir, analysis_type='parametric', aov=None):
    """Create type-specific analysis reports"""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    report_content = f"""
## Analysis Report: {dv} ({task_level})
### Data Transformations
Applied method: **{transforms.get(dv, 'none')}**
**Interpretation:** Transformations adjust data distribution to meet parametric assumptions. Yeo-Johnson preserves sign while handling zero/negative values.

### Assumption Checks
1. **DV Normality (Shapiro-Wilk)**
   - p-value: {assumptions['normality_pval']:.4f}
   - Passed: {'Yes' if assumptions['normality_passed'] else 'No'}
**Interpretation:** Tests if data follows normal distribution (null hypothesis assumes normal distribution). p > 0.05 does not reject null hypothesis."""

    if analysis_type == 'parametric':
        gg_applied = assumptions['epsilon_gg'] < 0.75
        hf_applied = 0.75 <= assumptions['epsilon_gg'] < 1
        correction = 'GG' if gg_applied else 'HF' if hf_applied else 'None'
        
        report_content += f"""
2. **Sphericity Handling**
   - Mauchly's p-value: {assumptions['sphericity_pval']:.4f}
   - Greenhouse-Geisser epsilon: {assumptions['epsilon_gg']:.3f}
   - Correction Applied: {correction}
**Interpretation:** Sphericity requires equal variance differences between conditions. epsilon < 1 indicates violation, requiring correction."""

        if aov is not None:
            if 'F' in aov.columns and 'ddof1' in aov.columns and 'ddof2' in aov.columns:
                f_val = aov['F'].values[0]
                df1 = aov['ddof1'].values[0]
                df2 = aov['ddof2'].values[0]
                eta_sq = (f_val * df1) / (f_val * df1 + df2)
                report_content += f"""
3. **Effect Size**
   - **Partial eta-squared:** {eta_sq:.3f}
**Interpretation:** Proportion of variance explained by factor."""

    eligibility_status = "Eligible for parametric analysis." if assumptions['valid'] else "Ineligible for parametric analysis."
    
    report_content += f"""\n
**Analysis Type:** {analysis_type.title()}
**Eligibility Status:** {eligibility_status}
**Note:** Parametric eligibility requires passed normality (p ≥ 0.05) and sphericity (p ≥ 0.05 or corrected)."""

    report_path = save_dir / f"{dv}_report.md"
    report_path.write_text(report_content.strip(), encoding="utf-8")

def analyze_task_data(data, task_level, variables):
    """Core analysis workflow with proper type separation"""
    parametric_dir = RESULTS_DIR / task_level / 'parametric'
    nonparametric_dir = RESULTS_DIR / task_level / 'nonparametric'
    parametric_dir.mkdir(parents=True, exist_ok=True)
    nonparametric_dir.mkdir(parents=True, exist_ok=True)
    
    filtered_vars = filter_dvs_by_missing(data, variables, task_level)
    
    parametric_results = []
    nonparametric_results = []
    transformations = {}
    
    # Add tracking variables
    parametric_candidates = []
    nonparametric_only = []

    for dv in filtered_vars:
        assumptions = check_rm_assumptions(data, dv)
        
        # Apply transformation if needed
        if not assumptions['valid'] and not assumptions['normality_passed']:
            pt = PowerTransformer(method='yeo-johnson')
            data[dv] = pt.fit_transform(data[[dv]])
            transformations[dv] = 'yeo-johnson'
            assumptions = check_rm_assumptions(data, dv)

        # Track analysis eligibility
        if assumptions['valid']:
            parametric_candidates.append(dv)
        else:
            nonparametric_only.append(dv)

        # Parametric analysis path
        if assumptions['valid']:
            try:
                aov = pg.rm_anova(
                    data=data,
                    dv=dv,
                    within='Env',
                    subject='Subject',
                    correction='auto'
                ).assign(DV=dv)
                
                # Reorder parametric columns
                parametric_cols = ['DV'] + \
                                [c for c in aov.columns if c.startswith('p-')] + \
                                [c for c in aov.columns if c not in ['DV'] and not c.startswith('p-')]
                aov = aov[parametric_cols]
                
                parametric_results.append(aov)
                generate_report(
                    assumptions=assumptions,
                    transforms=transformations,
                    dv=dv,
                    task_level=task_level,
                    save_dir=parametric_dir,
                    analysis_type='parametric',
                    aov=aov
                )
            except Exception as e:
                logging.error(f"Parametric analysis failed for {dv}: {str(e)}")

        # Non-parametric analysis path (ALWAYS RUN)
        try:
            friedman = pg.friedman(
                data=data,
                dv=dv,
                within='Env',
                subject='Subject'
            ).assign(DV=dv)
            
            # Reorder non-parametric columns
            nonparametric_cols = ['DV'] + \
                               [c for c in friedman.columns if c.startswith('p-')] + \
                               [c for c in friedman.columns if c not in ['DV'] and not c.startswith('p-')]
            friedman = friedman[nonparametric_cols]
            
            nonparametric_results.append(friedman)
            
            if not assumptions['valid']:
                generate_report(
                    assumptions=assumptions,
                    transforms=transformations,
                    dv=dv,
                    task_level=task_level,
                    save_dir=nonparametric_dir,
                    analysis_type='nonparametric'
                )
        except Exception as e:
            logging.error(f"Non-parametric analysis failed for {dv}: {str(e)}")

    # Save results with proper column order
    if parametric_results:
        pd.concat(parametric_results).to_csv(parametric_dir/'analysis_parametric.csv', index=False)
    
    if nonparametric_results:
        pd.concat(nonparametric_results).to_csv(nonparametric_dir/'analysis_nonparametric.csv', index=False)
    
    # Add logging statements
    if parametric_candidates:
        logging.info(f"{task_level} variables eligible for both analyses: {', '.join(parametric_candidates)}")
    
    if nonparametric_only:
        logging.info(f"{task_level} variables requiring non-parametric analysis: {', '.join(nonparametric_only)}")

    return filtered_vars

def generate_combined_visualizations(combined_data, dvs):
    """Create unified visualizations"""
    combined_dir = RESULTS_DIR / 'combined_visualizations'
    combined_dir.mkdir(parents=True, exist_ok=True)

    for dv in dvs:
        plt.figure(figsize=(10, 6), dpi=300)
        ax = sns.lineplot(
            data=combined_data,
            x='Env',
            y=dv,
            hue='Task',
            style='Task',
            markers=['o', 's'],
            dashes=False,
            errorbar=('ci', 95),
            palette='colorblind'
        )

        # Move legend to lower right corner with proper styling
        plt.legend(
            loc='lower right',
            bbox_to_anchor=(1.0, 0.0),
            frameon=True,
            framealpha=0.9,
            edgecolor='black'
        )

        plt.title(f"{dv} Performance Across Environments", fontsize=14, pad=20)
        plt.ylabel(dv.replace('_', ' ').title(), fontsize=12)
        plt.xlabel('Environment', fontsize=12)
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)
        
        plt.savefig(combined_dir / f"combined_{dv}.png", bbox_inches='tight', dpi=300)
        plt.close()

def main():
    """Orchestrates statistical analysis workflow for N-back task data.    
    Workflow steps:
    1. Initializes directory structure
    2. Loads and filters experimental data
    3. Performs parametric/non-parametric analyses
    4. Generates reports and visualizations
    """
    try:
        initialize_directories()
        raw_data = load_dataset()
        target_vars = [
            'NasaCLI', 'FTCLI_corrected', 'FTPACLI_corrected',
            'EngI_corrected', 'F1', 'TotalPress', 'IncorrNo', 
            'CorrRatio', 'RTCorrAv', 'RTIncorrAv'
        ]
        
        all_viz_data = []
        filtered_vars_per_task = {}

        for task in [2, 3]:
            task_level = f"{task}back"
            
            # Create separate copies for analysis and visualization
            task_data_raw = raw_data[raw_data['N'] == task].copy()  # Raw data for visualization
            task_data_analysis = task_data_raw.copy()  # Copy for analysis
            
            current_vars = [v for v in target_vars if v in task_data_analysis.columns]
            
            filtered_vars = analyze_task_data(task_data_analysis, task_level, current_vars)
            filtered_vars_per_task[task_level] = filtered_vars
            
            # Prepare visualization data from raw, unmodified data
            viz_data = task_data_raw[['Subject', 'Env'] + filtered_vars].copy()
            viz_data['Task'] = task_level
            all_viz_data.append(viz_data)
            logging.info(f"{task_level} variables: {', '.join(filtered_vars)}")

        if all_viz_data:
            common_dvs = set.intersection(*[set(vars) for vars in filtered_vars_per_task.values()])
            
            if common_dvs:
                logging.info(f"Common variables in 2-back and 3-back conditions: {', '.join(common_dvs)}")
                combined_data = pd.concat(all_viz_data)
                generate_combined_visualizations(combined_data, list(common_dvs))
            else:
                logging.warning("No common variables across tasks - generating task-specific plots")
                for task_level, vars in filtered_vars_per_task.items():
                    task_data = [d for d in all_viz_data if d['Task'].iloc[0] == task_level][0]
                    generate_combined_visualizations(task_data, vars)

    except Exception as e:
        logging.critical(f"Pipeline failure: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
