import warnings
warnings.filterwarnings(action="ignore", category=FutureWarning)
warnings.filterwarnings(action='ignore', message='invalid value encountered in power', category=RuntimeWarning, module='pingouin.multivariate')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import sys
import pingouin as pg
from pathlib import Path
from statsmodels.multivariate.manova import MANOVA
from sklearn.preprocessing import PowerTransformer
from scipy.stats import rankdata, boxcox, anderson
from pingouin import sphericity
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Configuration
BASE_DIR = Path(__file__).parent
DATA_PATH = BASE_DIR / 'results' / 'final_dataset.csv'
RESULTS_DIR = BASE_DIR / 'results' / 'statistical_analysis'
LOG_FILE = RESULTS_DIR / 'analysis.log'

RESULTS_DIR.mkdir(parents=True, exist_ok=True)  # Ensure directory exists for log

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler(sys.stdout)
    ]
)

def initialize_directories():
    """Create required output directories"""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    for subdir in ['assumptions', 'parametric_MANOVA', 'nonparametric_ANOVA', 'visualizations']:
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


def check_manova_assumptions(data, dvs, group_var='Subject'):
    """Checks MANOVA assumptions for repeated measures data."""
    assumptions = {
        'valid_dvs': [],
        'matrix_rank': None,
        'condition_number': None,
        'multivariate_normality': None,
        'sphericity': {},
        'correlation_check': {dv: [] for dv in dvs}
    }
    
    # 1. Matrix Rank Check (handle missing data properly)
    try:
        wide_data = data.pivot_table(index=group_var, columns=['Env', 'N'], values=dvs)
        wide_data = wide_data.dropna(axis=1, how='all').fillna(0)  # Improved handling
        assumptions['matrix_rank'] = np.linalg.matrix_rank(wide_data)
    except Exception as e:
        logging.error(f"Matrix rank check failed: {str(e)}")
        assumptions['matrix_rank'] = 0

    # 2. Multicollinearity Check (use correlation matrix)
    corr_matrix = data[dvs].corr().values
    assumptions['condition_number'] = np.linalg.cond(corr_matrix)  # More standard approach

    # 3. Multivariate Normality (Henze-Zirkler Test)
    try:
        hz_test = pg.multivariate_normality(data[dvs], alpha=0.05)  # Standard alpha
        assumptions['multivariate_normality'] = hz_test.pval >= 0.05
    except Exception as e:
        logging.error(f"Multivariate normality check failed: {str(e)}")
        assumptions['multivariate_normality'] = False

    # 4. Sphericity Checks (Mauchly's Test)
    for effect in [['Env'], ['N'], ['Env', 'N']]:
        effect_name = 'x'.join(effect)
        try:
            spher, W, chi2, dof, mauchly_p = pg.sphericity(data, dv=dvs[0], subject=group_var, within=effect)
            assumptions['sphericity'][effect_name] = {
                'pval': mauchly_p,
                'W': W,
                'chi2': chi2,
                'dof': dof
            }
        except Exception as e:
            logging.error(f"Sphericity check for {effect_name} failed: {str(e)}")
            assumptions['sphericity'][effect_name] = {'pval': 0.0}  # Store as dict

    # 5. Correlation Check (fixed and validated)
    corr_matrix = data[dvs].corr().abs()
    high_corr_pairs = []
    for i in range(len(dvs)):
        for j in range(i+1, len(dvs)):
            if corr_matrix.iloc[i, j] > 0.8:
                high_corr_pairs.append((dvs[i], dvs[j]))
    for dv1, dv2 in high_corr_pairs:
        assumptions['correlation_check'][dv1].append(dv2)
        assumptions['correlation_check'][dv2].append(dv1)

    # Final Validation (All DVs Together)
    assumptions['valid_dvs'] = dvs if (
        assumptions['multivariate_normality'] and
        all(v['pval'] >= 0.01 for v in assumptions['sphericity'].values()) and  # Access 'pval'
        assumptions['condition_number'] < 30
    ) else []

    return assumptions


def apply_transformations(data, dvs):
    """Apply and validate data transformations"""
    transformed = data.copy()
    applied = {}
    boxcox_lambdas = {}
    ad_stats = {}
    ad_criticals = {}

    for dv in dvs:
        original = transformed[dv]
        if (original <= 0).any():
            shifted = original + abs(original.min()) + 1e-6
            logging.info(f"Applied value shift to {dv}")
        else:
            shifted = original.copy()

        transforms = {}
        yeo = PowerTransformer(method='yeo-johnson').fit_transform(shifted.values.reshape(-1,1))
        transforms['yeo_johnson'] = yeo.flatten()
        try:
            boxcox_trans, lambda_ = boxcox(shifted)
            transforms['boxcox'] = boxcox_trans
            boxcox_lambdas[dv] = lambda_
        except Exception as e:
            logging.info(f"Box-Cox not applied to {dv}: {e}")        

        best_transform = 'original'
        best_stat = anderson(original).statistic
        best_critical = anderson(original).critical_values[2]  # 5% level

        for name, trans in transforms.items():
            result = anderson(trans)
            stat = result.statistic
            critical = result.critical_values[2]
            if stat < best_stat:
                best_stat = stat
                best_transform = name
                best_critical = critical

        if best_transform != 'original':
            transformed[dv] = transforms[best_transform]
            logging.info(f"Applied {best_transform} transform to {dv} (stat={best_stat:.4f})")
            applied[dv] = best_transform
        else:
            logging.info(f"No effective transform found for {dv}")
            applied[dv] = 'original'

        ad_stats[dv] = best_stat
        ad_criticals[dv] = best_critical

    return transformed, applied, boxcox_lambdas, ad_stats, ad_criticals

def run_parametric_manova(data, dvs):
    """Regularized MANOVA for repeated measures"""
    try:
        wide_data = data.pivot_table(
            index='Subject',
            columns=['Env', 'N'],
            values=dvs
        ).fillna(0)
        
        # Regularize with pseudo-inverse
        manova = MANOVA(wide_data, method='pinv')
        mv_test = manova.mv_test()
        
        # Extract and save results
        results = []
        for effect in mv_test.results:
            pillais = mv_test.results[effect]['stat'].loc["Pillai's trace"]
            results.append({
                'Effect': ':'.join(effect) if isinstance(effect, tuple) else str(effect),
                'Pillai\'s Trace': pillais['stat'],
                'F': pillais['F value'],
                'df_num': pillais['num df'],
                'df_den': pillais['den df'],
                'p_value': pillais['p value'],
            })
        
        results_df = pd.DataFrame(results)
        results_df.to_csv(RESULTS_DIR / 'parametric_MANOVA' / 'manova_results.csv', index=False, encoding='utf-8')
        
        # Generate formatted summary
        summary = [
            "# MANOVA Results",
            "## Significant Effects (Pillai's Trace)",
            "| Effect | Value | F | df | p-value |",
            "|--------|-------|---|---|---------|"
        ]
        
        for _, row in results_df.iterrows():
            sig = ("***" if row['p_value'] < 0.001 
                   else "**" if row['p_value'] < 0.01 
                   else "*" if row['p_value'] < 0.05 
                   else "")
            summary.append(
                f"| {row['Effect']} | {row['Pillai\'s Trace']:.3f} | "
                f"{row['F']:.2f} | {row['df_num']},{row['df_den']} | "
                f"{row['p_value']:.4f}{sig} |"
            )
        
        (RESULTS_DIR / 'parametric_MANOVA' / 'manova_summary.md').write_text('\n'.join(summary), encoding='utf-8')
        return True
        
    except Exception as e:
        logging.error(f"MANOVA execution failed: {str(e)}")
        return False


def run_nonparametric_analysis(data, dvs):
    """Aligned Rank Transform ANOVA"""
    try:
        results = []
        for dv in dvs:
            df = data.copy()
            # Remove parametric assumption checks
            aligned = df[dv] - df.groupby('Subject')[dv].transform('mean')
            df['ranked'] = rankdata(aligned)
            model = pg.rm_anova(data=df, dv='ranked', within=['Env','N'], subject='Subject')
            # Add DV column
            model['DV'] = dv 
            results.append(model)
        # Concatenate and reset index before saving
        pd.concat(results).reset_index(drop=True).to_csv(RESULTS_DIR / 'nonparametric_ANOVA' / 'art_ANOVA_results.csv', index=False, encoding='utf-8')
        return True
    except Exception as e:
        logging.error(f"ART ANOVA failed: {str(e)}")
        return False
    
def post_hoc_analysis(data):
    """Robust post-hoc analysis with empty result handling"""
    try:
        art_results = pd.read_csv(RESULTS_DIR / 'nonparametric_ANOVA' / 'art_ANOVA_results.csv')
        posthoc_dfs = []
        required_cols = ['Effect', 'A', 'B', 'T', 'dof', 'p-corr', 'hedges']

        for _, row in art_results[art_results['p_selected'] < 0.05].iterrows():
            try:
                comparisons = pg.pairwise_tests(
                    data=data, 
                    dv=row['DV'],
                    within=row['Source'].split('*'),
                    subject='Subject',
                    padjust='holm'
                ).assign(DV=row['DV'], Effect=row['Source'])
                
                if all(col in comparisons.columns for col in required_cols):
                    posthoc_dfs.append(comparisons[required_cols])
                    
            except Exception as e:
                logging.warning(f"Skipping {row['Source']} on {row['DV']}: {str(e)}")

        if posthoc_dfs:
            pd.concat(posthoc_dfs).to_csv(RESULTS_DIR / 'nonparametric_ANOVA' / 'post_hoc_results.csv', index=False, encoding='utf-8')

    except Exception as e:
        logging.error(f"Post-hoc analysis aborted: {str(e)}")

def save_assumption_reports(assumptions, save_dir):
    """Saves MANOVA assumption reports in Markdown format for repeated measures."""
    save_dir.mkdir(parents=True, exist_ok=True)
    report = ["# MANOVA Assumption Validation Report\n"]

    # 1. Matrix Rank Check (updated reference)
    report.append("## Matrix Rank Check")
    report.append(f"- Required: {len(assumptions['correlation_check'])}")  # Number of DVs
    report.append(f"- Actual: {assumptions['matrix_rank']}")

    # 2. Multicollinearity Check
    report.append("\n## Multicollinearity Check")
    report.append(f"- Condition Number: {assumptions['condition_number']:.1f}")
    report.append(f"- Threshold: <30")
    report.append(f"- Status: {'PASS' if assumptions['condition_number'] < 30 else 'FAIL'}")

    # 3. Multivariate Normality
    report.append("\n## Multivariate Normality (Henze-Zirkler)")
    hz_status = "PASS" if assumptions['multivariate_normality'] else "FAIL"
    report.append(f"- Status: {hz_status}")

    # 4. Sphericity Checks
    report.append("\n## Sphericity (Mauchly's Test)")
    report.append("| Effect | p-value | Status |")
    report.append("|--------|---------|--------|")
    for effect, data in assumptions['sphericity'].items():
        pval = data['pval']  # Extract p-value from dict
        status = "PASS" if pval >= 0.01 else "FAIL"
        report.append(f"| {effect} | {pval:.4f} | {status} |")

    # 5. Correlation Check
    report.append("\n## High Correlation Check (r > 0.8)")
    report.append("| DV | Correlated With |")
    report.append("|----|-----------------|")
    for dv, correlates in assumptions['correlation_check'].items():
        correlates_str = ', '.join(correlates) if correlates else 'None'
        report.append(f"| {dv} | {correlates_str} |")

    # 6. DV Exclusion Reasons (updated)
    report.append("\n## DV Exclusion Reasons")
    report.append("| DV | Failed Assumptions |")
    report.append("|----|--------------------|")
    for dv in assumptions['correlation_check']:
        failed = []
        if not assumptions['multivariate_normality']:
            failed.append("Multivariate Normality")
        if any(v['pval'] < 0.01 for v in assumptions['sphericity'].values()):  # Fixed line
            failed.append("Sphericity")
        if assumptions['condition_number'] >= 30:
            failed.append("Multicollinearity")
        if len(assumptions['correlation_check'][dv]) > 0:
            failed.append("High Correlation")
        
        report.append(f"| {dv} | {', '.join(failed) if failed else 'All checks passed'} |")
    
    (save_dir / 'manova_assumptions.md').write_text('\n'.join(report), encoding='utf-8')




def generate_visualizations(data, dvs, analysis_type):
    """Generate analysis-specific visualizations"""
    viz_dir = RESULTS_DIR / 'visualizations' / analysis_type.replace(' ', '_')
    viz_dir.mkdir(exist_ok=True, parents=True)
    
    for dv in dvs:
        plt.figure(figsize=(10, 6))
        sns.boxplot(x='Env', y=dv, hue='N', data=data)
        plt.title(f"{analysis_type} - {dv} Distribution by Environment and Task Difficulty")
        plt.savefig(viz_dir / f"{dv}_distribution.png")
        plt.close()
        
def main():
    """Main analysis workflow with dynamic DV removal"""
    try:
        initialize_directories()
        data = load_dataset()
        initial_dvs = ['NasaCLI', 'FTCLI_corrected', 'FTPACLI_corrected',
                      'EngI_corrected', 'F1', 'TotalPress', 'IncorrNo', 
                      'CorrRatio', 'RTCorrAv', 'RTIncorrAv']
        dvs = [col for col in initial_dvs if col in data.columns]
        
        # ===== Initial Assumption Check =====
        logging.info("\n=== Applying Data Transformations ===")
        transformed_data, transforms, boxcox_lambdas, ad_stats, ad_criticals = apply_transformations(data, dvs)
        
        if len(dvs) >= 2:
            logging.info("\n=== Initial MANOVA Assumption Check ===")
            initial_assumptions = check_manova_assumptions(transformed_data, dvs)
            save_assumption_reports(initial_assumptions, RESULTS_DIR / 'assumptions' / 'initial_assumptions')
            
            # ===== Dynamic DV Removal =====
            clean_dvs = dvs.copy()
            
            # 1. Remove DVs correlated with F1 (preserve F1)
            if 'F1' in initial_assumptions['correlation_check']:
                f1_correlates = initial_assumptions['correlation_check']['F1']
                if f1_correlates:
                    logging.info(f"Removing F1-correlated DVs: {', '.join(f1_correlates)}")
                    clean_dvs = [dv for dv in clean_dvs if dv not in f1_correlates]
                    clean_dvs.append('F1')  # Ensure F1 is retained
            
            # 2. Remove other highly correlated DVs
            for dv in initial_assumptions['correlation_check']:
                if dv != 'F1' and initial_assumptions['correlation_check'][dv]:
                    logging.info(f"Removing {dv} due to high correlation")
                    clean_dvs = [x for x in clean_dvs if x != dv]
            
            # ===== Recheck Assumptions =====
            if len(clean_dvs) >= 2:
                logging.info("\n=== Rechecking Assumptions with Cleaned DVs ===")
                final_assumptions = check_manova_assumptions(transformed_data, clean_dvs)
                save_assumption_reports(final_assumptions, RESULTS_DIR / 'assumptions' / 'final_assumptions')
                manova_dvs = final_assumptions['valid_dvs']
            else:
                manova_dvs = []
            
        else:
            manova_dvs = []
        
        # ===== Nonparametric Path =====
        # Always use all DVs with original data
        art_dvs = dvs.copy()
        
        # ===== Run Analyses =====
        manova_success = False
        art_success = False
        
        # Parametric MANOVA
        if manova_dvs:
            logging.info(f"\n=== Running MANOVA on: {', '.join(manova_dvs)} ===")
            manova_success = run_parametric_manova(transformed_data, manova_dvs)
            generate_visualizations(transformed_data, manova_dvs, 'MANOVA')
        else:
            logging.warning("\n=== No DVs meet MANOVA assumptions - skipping parametric analysis ===")
        
        # Nonparametric ART ANOVA (always runs)
        logging.info(f"\n=== Running ART ANOVA on all DVs: {', '.join(art_dvs)} ===")
        art_success = run_nonparametric_analysis(data.copy(), art_dvs)
        generate_visualizations(data, art_dvs, 'ART ANOVA')
        
        # Post-hoc analyses
        if manova_success:
            logging.info("\n=== MANOVA Post-hoc Analysis ===")
            
        if art_success:
            logging.info("\n=== ART ANOVA Post-hoc Analysis ===")
            post_hoc_analysis(data.copy())

        
    except Exception as e:
        logging.critical(f"Critical pipeline error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
