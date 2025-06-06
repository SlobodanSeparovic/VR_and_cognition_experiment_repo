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

def filter_dvs_by_missing(data, dvs, max_missing=0.2):
    """Identify variables with acceptable missing data levels"""
    valid = []
    for dv in dvs:
        missing = data[dv].isna().mean()
        if missing <= max_missing:
            valid.append(dv)
        else:
            logging.warning(f"Excluding {dv} - {missing:.1%} missing values")
    return valid

def apply_transformations(data, dvs):
    """Apply optimal transformation to normalize each variable"""
    transformed = data.copy()
    methods = {}
    
    transformation_candidates = [
        ('yeo_johnson', lambda x: PowerTransformer(method='yeo-johnson').fit_transform(x.reshape(-1,1)).flatten()),
        ('boxcox', lambda x: boxcox(x)[0]),
        ('log', lambda x: np.log(x + 1e-6)),  # Add small constant to handle zeros
        ('sqrt', lambda x: np.sqrt(x + 0.5)),  # Add 0.5 to handle zero/negative values
        ('cube_root', lambda x: np.cbrt(x)),
        ('reciprocal', lambda x: 1/(x + 0.1)),  # Add 0.1 to avoid division by zero
        ('asinh', lambda x: np.arcsinh(x))
    ]
    
    for dv in dvs:
        original = transformed[dv].copy()
        best_method = 'original'
        best_score = np.inf
        
        # Handle negative values
        min_val = original.min()
        if min_val <= 0:
            shifted = original + abs(min_val) + 1e-6
        else:
            shifted = original.copy()

        for name, transform in transformation_candidates:
            try:
                # Apply transformation
                trans_data = transform(shifted) if name in ['yeo_johnson', 'boxcox'] else transform(original)
                
                # Calculate normality score
                score = anderson(trans_data).statistic
                
                # Update best method if better score
                if score < best_score:
                    best_score = score
                    best_method = name
                    
            except Exception as e:
                logging.debug(f"Transformation {name} failed for {dv}: {str(e)}")
                continue

        # Apply best transformation
        if best_method != 'original':
            transform_func = dict(transformation_candidates)[best_method]
            transformed[dv] = transform_func(shifted) if best_method in ['yeo_johnson', 'boxcox'] else transform_func(original)
            logging.info(f"Applied {best_method} transformation to {dv}")
        else:
            logging.info(f"No transformation needed for {dv}")
            
        methods[dv] = best_method

    return transformed, methods

def check_rm_assumptions(data, dv, group_var='Subject'):
    """Validate ANOVA assumptions for single dependent variable"""
    assumptions = {
        'normality': {},
        'sphericity': {'pval': 1.0, 'W': 1.0},
        'homogeneity': True,
        'valid': False,
        'env_data': {}, 
        'epsilon': 1.0
    }
    
    # Normality per environment
    for env in data['Env'].unique():
        env_series = data[data['Env'] == env][dv].dropna()
        assumptions['env_data'][env] = env_series
        
        try:
            result = pg.normality(env_series)
            assumptions['normality'][env] = result['pval'].values[0] >= 0.05
        except Exception as e:
            logging.error(f"Normality check failed for {dv}/{env}: {str(e)}")
            assumptions['normality'][env] = False

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

    # Homogeneity check
    try:
        levene = pg.homoscedasticity(data, dv=dv, group='Env')
        assumptions['homogeneity'] = levene['pval'].values[0] >= 0.05
    except Exception as e:
        logging.error(f"Homogeneity check failed: {str(e)}")
        assumptions['homogeneity'] = False

    # Get epsilon from ANOVA
    try:
        aov = pg.rm_anova(data=data, dv=dv, within='Env', subject=group_var, correction='auto')
        assumptions['epsilon'] = aov.at[0, 'eps']
    except Exception as e:
        logging.error(f"Epsilon check failed: {str(e)}")
        assumptions['epsilon'] = 0.5

    assumptions['valid'] = (
        all(assumptions['normality'].values()) and
        assumptions['sphericity']['pval'] >= 0.05 and
        assumptions['homogeneity']
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
    # Ensure save_dir is a directory, not a file path
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    report = [
        f"## Analysis Report: {dv} ({task}-back)",
        "### Data Transformations",
        f"Applied method: **{transforms.get(dv, 'none')}**",
        "\n### Assumption Checks",
        "#### Normality (Shapiro-Wilk)",
        "| Environment | n | Passed | p-value  |",
        "|-------------|---|--------|----------|"
    ]
    for env, passed in assumptions['normality'].items():
        env_data = assumptions['env_data'].get(env, pd.Series())
        n = len(env_data)
        try:
            pval = pg.normality(env_data)['pval'].values[0]
        except:
            pval = np.nan
        report.append(f"| {env} | {n} | {'Yes' if passed else 'No'} | {pval:.4f} |")
    report.extend([
        "\n#### Sphericity (Mauchly's Test)",
        f"- W = {assumptions['sphericity'].get('W', 0):.3f}",
        f"- p = {assumptions['sphericity'].get('pval', 0):.4f}",
        f"- Passed: {'Yes' if assumptions['sphericity'].get('pval', 0) >= 0.05 else 'No'}",
        "\n#### Variance Homogeneity (Levene's)",
        f"- Passed: {'Yes' if assumptions.get('homogeneity', False) else 'No'}",
        f"\n**Overall Validity:** {'Yes' if assumptions['valid'] else 'No'}"
    ])
    report_path = save_dir / f"{dv}_report.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)  # Ensure parent exists
    report_path.write_text("\n".join(report), encoding="utf-8")
    logging.info(f"Saved assumption report for {dv} at {report_path}")

    
def generate_visualizations(data, dvs, task_level_str):
    """Task-specific visualizations"""
    viz_dir = RESULTS_DIR / task_level_str / 'visualizations'
    viz_dir.mkdir(exist_ok=True)
    
    for dv in dvs:
        plt.figure(figsize=(10, 6))
        sns.lineplot(x='Env', y=dv, data=data, errorbar=('ci', 95), estimator='mean')
        plt.title(f"{task_level_str} - {dv} by Environment")
        plt.savefig(viz_dir / f"{dv}_lineplot.png")
        plt.close()


def analyze_task_data(data, task_level, variables):
    """Core analysis workflow for single task difficulty"""
    filtered_vars = filter_dvs_by_missing(data, variables)
    logging.info(f"\n=== Analyzing {task_level}-back Task ===")
    if not filtered_vars:
        logging.warning(f"No valid variables for {task_level}-back analysis")
        return

    transformed, methods = apply_transformations(data, filtered_vars)

    # Check assumptions for ALL variables
    all_valid = True
    assumptions_dict = {}
    for dv in filtered_vars:
        assumptions = check_rm_assumptions(transformed, dv)
        assumptions_dict[dv] = assumptions
        if not assumptions['valid']:
            all_valid = False

    # Generate reports for ALL DVs
    parametric_dir = RESULTS_DIR / task_level / 'parametric'
    for dv in filtered_vars:
        generate_report(
            assumptions_dict[dv],
            methods,
            dv,
            task_level.replace('back', ''),
            parametric_dir
        )
        
    # Initialize results list
    results = []

    # Unified analysis path
    if all_valid:
        analysis_type = 'parametric'
        output_dir = parametric_dir
        # Run parametric analysis
        for dv in filtered_vars:
            try:
                aov = pg.rm_anova(
                    data=transformed,
                    dv=dv,
                    within='Env',
                    subject='Subject',
                    correction='auto'
                )
                aov['DV'] = dv
                results.append(aov)
            except Exception as e:
                logging.error(f"ANOVA failed for {dv}: {str(e)}")
    else:
        analysis_type = 'nonparametric'
        output_dir = RESULTS_DIR / task_level / 'nonparametric'
        # Run non-parametric analysis
        for dv in filtered_vars:
            try:
                friedman = pg.friedman(data=data, dv=dv, within='Env', subject='Subject')
                friedman['DV'] = dv
                results.append(friedman)
            except Exception as e:
                logging.error(f"Friedman failed for {dv}: {str(e)}")

    # Save combined results
    if results:
        output_dir.mkdir(parents=True, exist_ok=True)
        pd.concat(results).to_csv(output_dir / 'combined_results.csv', index=False)

    # Visualizations
    generate_visualizations(transformed if analysis_type == 'parametric' else data, filtered_vars, task_level)

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
        
        # Process each task difficulty separately
        for task in [2, 3]:
            task_level = f"{task}back"
            task_data = raw_data[raw_data['N'] == task].copy()
            current_vars = [v for v in target_vars if v in task_data.columns]
            analyze_task_data(task_data, task_level, current_vars)
            
    except Exception as e:
        logging.critical(f"Pipeline failure: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()