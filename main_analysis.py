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
from statsmodels.formula.api import ols
from scipy.stats import rankdata, boxcox, anderson

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
        # Parametric
        (RESULTS_DIR / task / 'parametric').mkdir(parents=True, exist_ok=True)
        # Non-parametric
        (RESULTS_DIR / task / 'nonparametric').mkdir(parents=True, exist_ok=True)
        # Visualizations
        (RESULTS_DIR / task / 'visualizations').mkdir(parents=True, exist_ok=True)
    logging.info("Analysis directories initialized")


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


def apply_transformations(data, dvs, group_var='Subject'):
    """Apply transformations to optimize residual normality for RM-ANOVA"""
    transformed = data.copy()
    methods = {}
    
    transformation_candidates = [
        ('yeo_johnson', lambda x: PowerTransformer(method='yeo-johnson').fit_transform(x.reshape(-1,1)).flatten()),
        ('boxcox', lambda x: boxcox(x)[0]),
        ('log', lambda x: np.log(x + 1e-6)),
        ('sqrt', lambda x: np.sqrt(x + 0.5)),
        ('cube_root', lambda x: np.cbrt(x)),
        ('reciprocal', lambda x: 1/(x + 0.1)),
        ('asinh', lambda x: np.arcsinh(x)),
        ('original', lambda x: x)  # Baseline comparison
    ]
    
    for dv in dvs:
        original = transformed[dv].copy()
        best_method = 'original'
        best_score = np.inf
        
        for name, transform in transformation_candidates:
            try:
                # Apply transformation with consistent shifting
                if name in ['yeo_johnson', 'boxcox']:
                    min_val = original.min()
                    shifted = original + abs(min_val) + 1e-6 if min_val <= 0 else original.copy()
                    trans_data = transform(shifted)
                else:
                    trans_data = transform(original)

                # Create temporary dataset
                temp_df = data[[group_var, 'Env']].copy()
                temp_df[dv] = trans_data
                
                # Fit repeated measures model
                model = ols(f"{dv} ~ C(Env) + C({group_var})", data=temp_df).fit()
                residuals = model.resid
                
                # Skip if insufficient data
                if len(residuals) < 3:
                    continue
                
                # Evaluate residual normality
                score = anderson(residuals).statistic
                
                if score < best_score:
                    best_score = score
                    best_method = name

            except Exception as e:
                logging.debug(f"Transformation {name} failed for {dv}: {str(e)}")
                continue

        # Apply best transformation consistently
        if best_method != 'original':
            if best_method in ['yeo_johnson', 'boxcox']:
                min_val = original.min()
                shifted = original + abs(min_val) + 1e-6 if min_val <= 0 else original.copy()
                transformed[dv] = transform(shifted)
            else:
                transformed[dv] = transform(original)
                
            logging.info(f"Applied {best_method} transformation to {dv}")
            
        methods[dv] = best_method

    return transformed, methods


def check_rm_assumptions(data, dv, group_var='Subject'):
    """Validate ANOVA assumptions for single dependent variable"""
    assumptions = {
        'residual_normality': False,
        'sphericity': {'pval': 1.0, 'W': 1.0},
        'valid': False
    }

    # Residual normality check
    try:
        # Get residuals
        # Fit the model using statsmodels
        model = ols(f"{dv} ~ C(Env) + C({group_var})", data=data).fit()
        # Get residuals
        residuals = model.resid
        # test the residuals for normality
        norm_test = pg.normality(residuals)
        assumptions['residual_normality'] = norm_test['pval'].values[0] >= 0.05
    except Exception as e:
        logging.error(f"Residual normality check failed: {str(e)}")
        assumptions['residual_normality'] = False

    # Sphericity check
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sph = pg.sphericity(data, dv=dv, subject=group_var, within=['Env'])
            assumptions['sphericity']['pval'] = sph[4]
            assumptions['sphericity']['W'] = sph[1]
    except Exception as e:
        logging.error(f"Sphericity check failed: {str(e)}")
        assumptions['sphericity']['pval'] = 0.0

    # Validity condition
    assumptions['valid'] = (
        assumptions['residual_normality'] and 
        assumptions['sphericity']['pval'] >= 0.05
    )
    
    return assumptions

def run_parametric_analysis(data, dv):
    """Execute repeated measures ANOVA"""
    try:
        return pg.rm_anova(data=data, dv=dv, within='Env', subject='Subject')
    except Exception as e:
        logging.error(f"ANOVA failed for {dv}: {str(e)}")
        return None

def run_nonparametric_analysis(data, dv):
    """Perform Friedman test as non-parametric alternative"""
    try:
        return pg.friedman(data=data, dv=dv, within='Env', subject='Subject')
    except Exception as e:
        logging.error(f"Friedman test failed for {dv}: {str(e)}")
        return None

def generate_report(assumptions, transforms, dv, task, save_dir):
    """Create detailed Markdown report for each analysis"""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    report = [
        f"## Analysis Report: {dv} ({task}-back)",
        "### Data Transformations",
        f"Applied method: **{transforms.get('method', 'none')}**",
        "\n### Assumption Checks",
        "#### Residual Normality (Shapiro-Wilk)",
        f"- p-value: {pg.normality(assumptions.get('residuals', [0]))['pval'].values[0]:.4f}",
        f"- Passed: {'Yes' if assumptions['residual_normality'] else 'No'}",
        "\n#### Sphericity (Mauchly's Test)",
        f"- W = {assumptions['sphericity'].get('W', 0):.3f}",
        f"- p-value = {assumptions['sphericity'].get('pval', 0):.4f}",
        f"- Passed: {'Yes' if assumptions['sphericity'].get('pval', 0) >= 0.05 else 'No'}",
        f"\n**Overall Validity:** {'Yes' if assumptions['valid'] else 'No'}"
    ]
    
    report_path = save_dir / f"{dv}_report.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(report), encoding="utf-8")
    logging.info(f"Saved assumption report for {dv} at {report_path}")

    
def generate_visualizations(data, dvs, task_level_str):
    """Task-specific visualizations"""
    viz_dir = RESULTS_DIR / task_level_str / 'visualizations'
    viz_dir.mkdir(exist_ok=True)
    
    # Plot only requested DVs
    for dv in dvs:
        if dv not in data.columns:
            logging.warning(f"Skipping visualization for {dv} - column missing")
            continue
            
        plt.figure(figsize=(10, 6))
        sns.lineplot(x='Env', y=dv, data=data, errorbar=('ci', 95), estimator='mean')
        plt.title(f"{task_level_str} - {dv} by Environment")
        plt.savefig(viz_dir / f"{dv}_lineplot.png", bbox_inches='tight')
        plt.close()


def analyze_task_data(data, task_level, variables):
    """Core analysis workflow for single task difficulty"""
    logging.info(f"\n=== Analyzing {task_level}-back Task ===")
    
    # Filter data
    filtered_vars = filter_dvs_by_missing(data, variables, task_level)
    if not filtered_vars:
        logging.warning(f"No valid variables for {task_level}-back analysis")
        return []

    # Initialize results lists
    parametric_results = []
    nonparametric_results = []
    assumptions_dict = {}
    all_valid = True

    # Check assumptions for all variables
    for dv in filtered_vars:
        # First check assumptions
        assumptions = check_rm_assumptions(data, dv)
        assumptions_dict[dv] = assumptions
        
        # If assumptions are not valid, apply transformations
        if not assumptions['valid']:
            logging.info(f"Assumptions not met for {dv}. Applying transformations.")
            transformed, methods = apply_transformations(data, [dv],)
            # Recheck assumptions on transformed data
            assumptions = check_rm_assumptions(transformed, dv)
            assumptions_dict[dv] = assumptions
            
            if not assumptions['valid']:
                all_valid = False
                logging.warning(f"Assumptions still not met for {dv} after transformation.")
            else:
                logging.info(f"Assumptions met for {dv} after transformation.")
        else:
            logging.info(f"Assumptions met for {dv}.")

    # Define output directories
    parametric_dir = RESULTS_DIR / task_level / 'parametric'
    nonparametric_dir = RESULTS_DIR / task_level / 'nonparametric'
    
    # Run non-parametric analysis for all DVs
    for dv in filtered_vars:
        try:
            friedman = pg.friedman(data=data, dv=dv, within='Env', subject='Subject')
            friedman['DV'] = dv
            nonparametric_results.append(friedman)
        except Exception as e:
            logging.error(f"Friedman failed for {dv}: {str(e)}")

    # Run parametric analysis for valid DVs
    for dv in filtered_vars:
        if assumptions_dict[dv]['valid']:
            try:
                aov = pg.rm_anova(
                    data=transformed if not assumptions_dict[dv]['valid'] else data,
                    dv=dv,
                    within='Env',
                    subject='Subject',
                    correction='auto'
                )
                aov['DV'] = dv
                parametric_results.append(aov)
            except Exception as e:
                logging.error(f"ANOVA failed for {dv}: {str(e)}")

    # Save combined results for non-parametric analysis
    if nonparametric_results:
        nonparametric_dir.mkdir(parents=True, exist_ok=True)
        pd.concat(nonparametric_results).to_csv(nonparametric_dir / 'combined_results.csv', index=False)

    # Save combined results for parametric analysis
    if parametric_results:
        parametric_dir.mkdir(parents=True, exist_ok=True)
        pd.concat(parametric_results).to_csv(parametric_dir / 'combined_results.csv', index=False)

    # Prepare visualization data
    viz_data = transformed[['Subject', 'Env'] + filtered_vars].copy() if any(assumptions_dict[dv]['valid'] for dv in filtered_vars) else data[['Subject', 'Env'] + filtered_vars].copy()

    # Visualizations
    generate_visualizations(viz_data, filtered_vars, task_level)
    
    return filtered_vars
    
def generate_combined_visualizations(combined_data, dvs):
    """Create combined 2-back vs 3-back visualizations"""
    combined_dir = RESULTS_DIR / 'combined_visualizations'
    combined_dir.mkdir(parents=True, exist_ok=True)
    
    for dv in dvs:
        plt.figure(figsize=(12, 8))
        sns.lineplot(
            data=combined_data,
            x='Env',
            y=dv,
            hue='Task',
            style='Task',
            markers=True,
            errorbar=('ci', 95),
            estimator='mean'
        )
        plt.title(f"{dv} Comparison: 2-back vs 3-back Tasks")
        plt.ylabel(dv)
        plt.savefig(combined_dir / f"combined_{dv}_comparison.png", bbox_inches='tight')
        plt.close()


def main():
    """Main analysis workflow controller"""
    try:
        initialize_directories()
        raw_data = load_dataset()
        target_vars = [
            'NasaCLI', 'FTCLI_corrected', 'FTPACLI_corrected',
            'EngI_corrected', 'F1', 'TotalPress', 'IncorrNo', 
            'CorrRatio', 'RTCorrAv', 'RTIncorrAv'
        ]
        
        # Store filtered variables and raw data for each task
        filtered_vars_per_task = {}
        all_raw = []
        
        # Process each task difficulty separately
        for task in [2, 3]:
            task_level = f"{task}back"
            task_data = raw_data[raw_data['N'] == task].copy()
            current_vars = [v for v in target_vars if v in task_data.columns]
            
            # Run analysis and get filtered variables
            filtered_vars = analyze_task_data(task_data, task_level, current_vars)
            filtered_vars_per_task[task_level] = filtered_vars
            
            # Prepare visualization data with only filtered variables
            viz_data = task_data[['Subject', 'Env'] + filtered_vars].copy()
            viz_data['Task'] = task_level
            all_raw.append(viz_data)

        # Generate combined visualizations using common filtered DVs
        if all_raw:
            # Get variables present in ALL tasks after filtering
            common_dvs = set.intersection(*[set(vars) for vars in filtered_vars_per_task.values()])
            
            if common_dvs:
                combined_data = pd.concat(all_raw)
                generate_combined_visualizations(combined_data, list(common_dvs))
            else:
                logging.warning("No common variables across tasks for combined visualization")

    except Exception as e:
        logging.critical(f"Pipeline failure: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()