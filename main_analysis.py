import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import sys
import pingouin as pg
from pathlib import Path
from scipy import stats
from statsmodels.multivariate.manova import MANOVA
from scipy.stats import rankdata

# Configuration
BASE_DIR = Path(__file__).parent
DATA_PATH = BASE_DIR / 'results' / 'final_dataset.csv'
RESULTS_DIR = BASE_DIR / 'results' / 'MANOVA'
LOG_FILE = RESULTS_DIR / 'analysis.log'

def initialize_directories():
    """Create required output directories"""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    for subdir in ['assumptions', 'parametric', 'nonparametric', 'visualizations']:
        (RESULTS_DIR / subdir).mkdir(exist_ok=True)
    logging.info("Analysis directories initialized")

def load_dataset():
    """Load and validate input data"""
    try:
        df = pd.read_csv(DATA_PATH)
        logging.info(f"Data loaded successfully from {DATA_PATH}")
    except FileNotFoundError:
        logging.critical(f"Data file not found: {DATA_PATH}")
        sys.exit(1)
        
    # Remove columns with >20% missing values
    missing_cols = df.columns[df.isnull().mean() > 0.2].tolist()
    if missing_cols:
        logging.warning(f"Removing columns with excessive missing values: {missing_cols}")
        df = df.drop(columns=missing_cols)
    
    return df

def check_repeated_measures_assumptions(data, dvs):
    """Check of RM-MANOVA assumptions"""
    assumptions = {
        'normality': {},
        'multicollinearity': None
    }
    
    # Separate indices
    subjective_dvs = ['NasaCLI']  # Protected from removal
    physiological_dvs = [dv for dv in dvs if dv not in subjective_dvs]
    
    # 1. Normality checks (Shapiro-Wilk per group)
    logging.info("\nChecking normality assumptions...")
    for dv in dvs:
        for env in data['Env'].unique():
            for n in data['N'].unique():
                subset = data[(data['Env'] == env) & (data['N'] == n)][dv]
                _, p = stats.shapiro(subset)
                assumptions['normality'][(dv, env, n)] = p
                if p < 0.05:
                    logging.warning(f"Normality violated: {dv} in {env}/N{n} (p={p:.4f})")

    # 2. Multicollinearity check (only for physiological indices)
    if physiological_dvs:
        corr_matrix = data[physiological_dvs].corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop = [col for col in upper.columns if any(upper[col] > 0.7)]
    else:
        to_drop = []
    
    assumptions['multicollinearity'] = to_drop

    # Save assumption checks
    pd.DataFrame(assumptions['normality'], index=['p-value']).T.to_csv(
        RESULTS_DIR / 'assumptions' / 'normality_check.csv')
    corr_matrix.to_csv(RESULTS_DIR / 'assumptions' / 'correlation_matrix.csv')
    
    return assumptions

def apply_transformations(data, dvs):
    """Apply and validate data transformations"""
    transformed = data.copy()
    applied = {}
    
    for dv in dvs:
        original = transformed[dv]
        
        # Handle non-positive values
        if (original <= 0).any():
            shifted = original + abs(original.min()) + 1e-6
            logging.info(f"Applied value shift to {dv}")
        else:
            shifted = original.copy()
        
        # Test transformation effectiveness
        transforms = {
            'log': np.log(shifted),
            'sqrt': np.sqrt(shifted),
            'cube_root': np.cbrt(shifted),
            'original': original
        }
        
        best_transform = 'original'
        best_p = stats.shapiro(original)[1]
        
        for name, trans in transforms.items():
            if name == 'original': continue
            _, p = stats.shapiro(trans)
            if p > best_p:
                best_p = p
                best_transform = name
                transformed[dv] = trans
        
        if best_transform != 'original':
            logging.info(f"Applied {best_transform} transform to {dv} (p={best_p:.4f})")
            applied[dv] = best_transform
        else:
            logging.info(f"No effective transform found for {dv}")
    
    return transformed, applied

def run_parametric_manova(data, dvs):
    """RM-MANOVA using Pillai's Trace"""
    try:
        results = []
        # Single loop through DVs
        for dv in dvs:
            # Pivot and validate data for current DV
            wide_data = data.pivot_table(index='Subject', 
                                        columns=['Env','N'], 
                                        values=dv)
            
            if wide_data.empty:
                logging.warning(f"Skipping {dv} - insufficient data")
                continue

            # Run MANOVA and extract Pillai's Trace
            manova = MANOVA(wide_data)
            mv_test = manova.mv_test()
            
            # Get results for current DV
            pillais = mv_test.results['x0']['stat'].loc["Pillai's trace"]
            results.append({
                'DV': dv,
                'Pillai\'s Trace': pillais['stat'],
                'F': pillais['F value'],
                'df_num': pillais['num df'],
                'df_den': pillais['den df'],
                'p_value': pillais['p value'],
            })
        
        # Save all results
        pd.DataFrame(results).to_csv(RESULTS_DIR / 'parametric' / 'pillais_results.csv', index=False)
        return len(results) > 0
        
    except Exception as e:
        logging.error(f"MANOVA failed: {str(e)}")
        return False

def run_nonparametric_analysis(data, dvs):
    """Aligned Rank Transform ANOVA"""
    try:
        results = []
        for dv in dvs:
            # Create copy to avoid modifying original data
            df = data.copy()
            
            # Align and rank data
            aligned = df[dv] - df.groupby('Subject')[dv].transform('mean')
            df['ranked'] = rankdata(aligned)
            
            # Fit RM-ANOVA to ranks
            model = pg.rm_anova(data=df, dv='ranked', 
                               within=['Env','N'], 
                               subject='Subject',
                               detailed=True)
            results.append(model)
        
        # Save results
        pd.concat(results).to_csv(RESULTS_DIR / 'nonparametric' / 'art_results.csv')
        return True
    except Exception as e:
        logging.error(f"ART ANOVA failed: {str(e)}")
        return False


def generate_visualizations(data, dvs):
    """Create analysis visualizations"""
    logging.info("\nGenerating diagnostic visualizations...")
    
    # Boxplots for each DV
    for dv in dvs:
        plt.figure(figsize=(10, 6))
        sns.boxplot(x='Env', y=dv, hue='N', data=data)
        plt.title(f"Distribution of {dv} by Condition")
        plt.savefig(RESULTS_DIR / 'visualizations' / f'{dv}_distribution.png', dpi=300)
        plt.close()
    
    # Subject Trend Lineplots
    for dv in dvs:
        plt.figure(figsize=(10, 6))
        sns.lineplot(x='Env', y=dv, hue='Subject', 
                    data=data, legend=False, alpha=0.3)
        plt.title(f"{dv} Subject Trends by Environment")
        plt.savefig(RESULTS_DIR / 'visualizations' / f'{dv}_subject_trends.png', dpi=300)
        plt.close()
    
    # Interaction plots
    for dv in dvs:
        plt.figure(figsize=(10, 6))
        sns.pointplot(x='Env', y=dv, hue='N', data=data, errorbar=('ci', 95))  # Fixed here
        plt.title(f"{dv} Interaction Effects")
        plt.savefig(RESULTS_DIR / 'visualizations' / f'{dv}_interaction.png', dpi=300)
        plt.close()
        
def main():
    """Main analysis workflow"""
    try:
        # Create directories
        initialize_directories()        
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(LOG_FILE),
                logging.StreamHandler(sys.stdout)
            ]
        )
        data = load_dataset()
        initial_dvs = ['NasaCLI', 'FTCLI_corrected', 'FTPACLI_corrected',
                      'EngI_corrected', 'F1', 'TotalPress', 'IncorrNo', 'CorrRatio', 'RTCorrAv', 'RTIncorrAv']
        dvs = [col for col in initial_dvs if col in data.columns]
        
        # Initial assumption check
        logging.info("\n=== Initial Assumption Check ===")
        assumptions = check_repeated_measures_assumptions(data, dvs)
        
        # Handle multicollinearity (only physiological indices)
        physiological_dvs = [dv for dv in dvs if dv != 'NasaCLI']
        final_physiological = [dv for dv in physiological_dvs 
                              if dv not in assumptions['multicollinearity']]
        
        # Combine with protected NASA-TLX
        final_dvs = final_physiological + ['NasaCLI']
        
        if len(final_dvs) != len(dvs):
            logging.warning(f"Removed physiological variables: {list(set(dvs) - set(final_dvs))}")
        
        # Apply transformations if needed
        if not all(p >= 0.05 for p in assumptions['normality'].values()):
            logging.info("\n=== Applying Data Transformations ===")
            transformed_data, transforms = apply_transformations(data, final_dvs)
            logging.info("Transformations applied: " + ", ".join(
                [f"{k} ({v})" for k, v in transforms.items()]))
            
            # Recheck assumptions
            logging.info("\n=== Post-Transformation Assumption Check ===")
            assumptions = check_repeated_measures_assumptions(transformed_data, final_dvs)
        else:
            transformed_data = data.copy()
        
        # Determine analysis path
        parametric_viable = all(p >= 0.05 for p in assumptions['normality'].values())
        
        if parametric_viable:
            logging.info("\n=== Running Parametric Analysis ===")
            success = run_parametric_manova(transformed_data, final_dvs)
        else:
            logging.warning("Assumptions not met - switching to non-parametric")
            success = run_nonparametric_analysis(transformed_data, final_dvs)
        
        # Generate visualizations
        if success:
            generate_visualizations(transformed_data, final_dvs)
            logging.info("\nANALYSIS COMPLETED SUCCESSFULLY")
        else:
            logging.error("Analysis failed to complete")
            
    except Exception as e:
        logging.critical(f"Critical pipeline error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
