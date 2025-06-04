import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning, module="pingouin.multivariate", message="divide by zero encountered in log")
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
from scipy.stats import rankdata, boxcox
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

def check_manova_assumptions(data, dvs):
    """Check MANOVA-specific assumptions with numerical stability"""
    assumptions = {
        'multivariate_normality': None,
        'homogeneity_covariance': None,
        'valid_dvs': []
    }
    
    if len(dvs) < 2:
        return assumptions
    
    try:
        # Create assumptions directory
        (RESULTS_DIR / 'assumptions').mkdir(exist_ok=True)
        
        # Start Markdown report
        report = ["# MANOVA Assumption Checks\n"]
        
        # 1. Filter DVs with insufficient variation
        valid_dvs = [dv for dv in dvs if np.var(data[dv]) > 1e-6]
        report.append(f"## Initial DV Filtering\n- Removed DVs with variance ≤1e-6: {list(set(dvs)-set(valid_dvs)) or 'None'}")
        
        # 2. Check for rank deficiency
        X = data[valid_dvs].values
        matrix_rank = np.linalg.matrix_rank(X)
        if matrix_rank < len(valid_dvs):
            logging.warning("Rank-deficient matrix detected")
            report.append(f"## Rank Check\n- **FAILED**: Matrix rank ({matrix_rank}) < number of DVs ({len(valid_dvs)})")
            with open(RESULTS_DIR / 'assumptions' / 'manova_assumptions.md', 'w') as f:
                f.write('\n'.join(report))
            return assumptions
            
        # 3. Check covariance matrix conditioning
        cov_matrix = np.cov(X, rowvar=False)
        cond_number = np.linalg.cond(cov_matrix)
        report.append(f"## Covariance Matrix Check\n- Condition number: {cond_number:.2e}")
        
        if cond_number > 1e12:
            logging.warning(f"Ill-conditioned covariance matrix (cond={cond_number:.2e})")
            report.append("- **FAILED**: Condition number >1e12 (indicates numerical instability)")
            with open(RESULTS_DIR / 'assumptions' / 'manova_assumptions.md', 'w') as f:
                f.write('\n'.join(report))
            return assumptions

        # 4. Multivariate normality test
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            try:
                result = pg.multivariate_normality(X)
                assumptions['multivariate_normality'] = result.pval
                report.append(f"## Normality Check\n- Henze-Zirkler p-value: {result.pval:.4f}")
            except Exception as e:
                logging.error(f"Normality test failed: {str(e)}")
                report.append("## Normality Check\n- **FAILED**: Could not compute")
                assumptions['multivariate_normality'] = None

        # 5. Homogeneity of covariance matrices
        try:
            data = data.copy()
            data['group'] = data['Env'] + '_' + data['N'].astype(str)
            box_result = pg.box_m(data, dvs=valid_dvs, group='group')
            assumptions['homogeneity_covariance'] = box_result['pval'].item()
            report.append(f"## Covariance Homogeneity\n- Box's M p-value: {box_result['pval'].item():.4f}")
        except Exception as e:
            logging.warning(f"Box's M test failed: {str(e)}")
            report.append("## Covariance Homogeneity\n- **FAILED**: Could not compute")
            assumptions['homogeneity_covariance'] = None

        # Save full report
        assumptions['valid_dvs'] = valid_dvs
        report.append(f"\n## Final Eligibility\n- Valid DVs: {valid_dvs or 'None'}")
        
        with open(RESULTS_DIR / 'assumptions' / 'manova_assumptions.md', 'w', encoding='utf-8') as f:
            f.write('\n'.join(report))

    except Exception as e:
        logging.error(f"Assumption check failed: {str(e)}")
    
    return assumptions


def apply_transformations(data, dvs):
    """Apply and validate data transformations"""
    transformed = data.copy()
    applied = {}
    boxcox_lambdas = {}
    
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
        
        # Add Box-Cox only if all values are positive
        try:
            boxcox_trans, lambda_ = boxcox(shifted)
            transforms['boxcox'] = boxcox_trans
        except Exception as e:
            logging.info(f"Box-Cox not applied to {dv}: {e}")        
        
        best_transform = 'original'
        best_p = stats.shapiro(original)[1]
        
        for name, trans in transforms.items():
            if name == 'original':
                continue
            _, p = stats.shapiro(trans)
            if p > best_p:
                best_p = p
                best_transform = name
                transformed[dv] = trans
                if name == 'boxcox':
                    boxcox_lambdas[dv] = lambda_
        
        if best_transform != 'original':
            logging.info(f"Applied {best_transform} transform to {dv} (p={best_p:.4f})")
            applied[dv] = best_transform
        else:
            logging.info(f"No effective transform found for {dv}")
    
    return transformed, applied, boxcox_lambdas

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
        pd.DataFrame(results).to_csv(RESULTS_DIR / 'parametric_MANOVA' / 'pillais_results.csv', index=False)
        return len(results) > 0
        
    except Exception as e:
        logging.error(f"MANOVA failed: {str(e)}")
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
        pd.concat(results).reset_index(drop=True).to_csv(
            RESULTS_DIR / 'nonparametric_ANOVA' / 'art_ANOVA_results.csv', index=False)
        return True
    except Exception as e:
        logging.error(f"ART ANOVA failed: {str(e)}")
        return False
    
def post_hoc_analysis(data):
    """Perform and save post-hoc analysis for significant ART results"""
    try:
        # Load and prepare ART results
        art_path = RESULTS_DIR / 'nonparametric_ANOVA' / 'art_ANOVA_results.csv'
        art_results = pd.read_csv(art_path)
        
        # Determine p-value selection based on sphericity (GG corrected or uncorrected)
        sphericity_thresh = 0.75
        use_gg = art_results['eps'] < sphericity_thresh
        art_results['p_selected'] = np.where(use_gg, art_results['p-GG-corr'], art_results['p-unc'])
        art_results['p_used'] = np.where(use_gg, 'GG-corrected', 'Uncorrected')
        
        # Identify significant results and those approaching significance
        sig_thresh = 0.05
        approaching_thresh = 0.1
        sig_mask = art_results['p_selected'] < sig_thresh
        approaching_mask = art_results['p_selected'].between(sig_thresh, approaching_thresh)
        
        significant = art_results[sig_mask].copy()
        approaching = art_results[approaching_mask].copy()
        
        # Generate summary report
        summary = [
            "# Aligned Ranks Transformation (ART) ANOVA Results Summary",
            f"Significance thresholds: p < {sig_thresh} (significant), p < {approaching_thresh} (approaching)"
        ]
        
        if not significant.empty:
            summary.append("## Statistically Significant Findings")
            sig_display = significant[['Source', 'DV', 'F', 'p_selected', 'ng2', 'p_used']].rename(
                columns={
                    'p_selected': 'p',
                    'p_used': 'Correction',
                    'ng2': 'effect size ng2'
                }
            )
            summary.append(sig_display.to_markdown(index=False))
        
        if not approaching.empty:
            summary.append("## Findings Approaching Significance")
            appr_display = approaching[['Source', 'DV', 'F', 'p_selected', 'ng2', 'p_used']].rename(
                columns={
                    'p_selected': 'p',
                    'p_used': 'Correction',
                    'ng2': 'effect size ng2'
                }
            )
            summary.append(appr_display.to_markdown(index=False))
        
        # Post-hoc analysis
        posthoc_dfs = []
        for _, row in significant.iterrows():
            dv = row['DV']
            effect = row['Source']
            factors = effect.split(' * ') if '*' in effect else [effect]
            try:
                comparisons = pg.pairwise_tests(
                    data=data, dv=dv, within=factors, subject='Subject', padjust='holm'
                )
                comparisons['ART_p'] = row['p_selected']
                comparisons['Effect'] = effect
                comparisons['DV'] = dv
                posthoc_dfs.append(comparisons)
            except Exception as e:
                logging.warning(f"Failed posthoc for {effect} on {dv}: {str(e)}")
        
        # Save outputs
        if posthoc_dfs:
            pd.concat(posthoc_dfs).to_csv(
                RESULTS_DIR / 'nonparametric_ANOVA' / 'post_hoc_results.csv', 
                index=False
            )
        with open(RESULTS_DIR / 'nonparametric_ANOVA' / 'art_ANOVA_summary.md', 'w') as f:
            f.write('\n\n'.join(summary))
        logging.info(f"Post-hoc analysis completed. {len(significant)} significant effects found.")
        
        if posthoc_dfs:
            posthoc_df = pd.concat(posthoc_dfs)
            posthoc_df.to_csv(
                RESULTS_DIR / 'nonparametric_ANOVA' / 'post_hoc_results.csv', 
                index=False
            )

            # Generate post-hoc summary
            posthoc_summary = ["# Post-Hoc Pairwise Comparisons Summary"]

            # Filter for significant and approaching significance
            sig_mask = posthoc_df['p-corr'] < 0.05
            approaching_mask = posthoc_df['p-corr'].between(0.05, 0.1)
            significant = posthoc_df[sig_mask]
            approaching = posthoc_df[approaching_mask]

            if not significant.empty:
                posthoc_summary.append("## Statistically Significant Pairwise Comparisons")
                display_cols = ['Effect', 'DV', 'A', 'B', 'T', 'dof', 'p-corr', 'hedges']
                posthoc_summary.append(significant[display_cols].rename(
                    columns={
                        'A': 'Group 1',
                        'B': 'Group 2',
                        'T': 't-value',
                        'dof': 'df',
                        'p-corr': 'corrected p',
                        'hedges': 'Hedges\' g'
                    }
                ).to_markdown(index=False))

            if not approaching.empty:
                posthoc_summary.append("## Pairwise Comparisons Approaching Significance")
                posthoc_summary.append(approaching[display_cols].rename(
                    columns={
                        'A': 'Group 1',
                        'B': 'Group 2',
                        'T': 't-value',
                        'dof': 'df',
                        'p-corr': 'corrected p',
                        'hedges': 'Hedges\' g'
                    }
                ).to_markdown(index=False))

            with open(RESULTS_DIR / 'nonparametric_ANOVA' / 'post_hoc_summary.md', 'w') as f:
                f.write('\n\n'.join(posthoc_summary))
                
    except Exception as e:
        logging.error(f"Post-hoc analysis failed: {str(e)}")
        raise

def save_assumption_reports(manova_assumptions, manova_dvs, art_dvs):
    """Save detailed assumption reports with numerical data"""
    assumption_dir = RESULTS_DIR / 'assumptions'
    assumption_dir.mkdir(exist_ok=True)

    # Prepare the manova_dvs string
    if manova_dvs:
        manova_dvs_str = '\n'.join(f'- {dv}' for dv in manova_dvs)
    else:
        manova_dvs_str = '*None*'

    # Prepare the art_dvs string
    art_dvs_str = '\n'.join(f'- {dv}' for dv in art_dvs)

    # 1. Human-readable summary
    with open(assumption_dir / 'assumption_summary.md', 'w', encoding='utf-8') as f:
        f.write('\n\n'.join([
            "# Statistical Analysis Assumption Reports",
            "## MANOVA Eligibility",
            f"- **Multivariate normality p-value**: {manova_assumptions['multivariate_normality']:.4f}",
            f"- **Homogeneity of covariance p-value**: {manova_assumptions['homogeneity_covariance']:.4f}",
            "### Eligible DVs for MANOVA:",
            manova_dvs_str,
            "## ART ANOVA Coverage",
            "### All Analyzed DVs:",
            art_dvs_str
        ]))

    # 2. Save raw numerical data
    pd.DataFrame({
        'check': ['multivariate_normality', 'homogeneity_covariance'],
        'p_value': [manova_assumptions['multivariate_normality'], 
                   manova_assumptions['homogeneity_covariance']]
    }).to_csv(assumption_dir / 'manova_assumption_values.csv', index=False)

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
    """Main analysis workflow with parallel parametric/nonparametric paths"""
    try:
        initialize_directories()
        data = load_dataset()
        initial_dvs = ['NasaCLI', 'FTCLI_corrected', 'FTPACLI_corrected',
                      'EngI_corrected', 'F1', 'TotalPress', 'IncorrNo', 
                      'CorrRatio', 'RTCorrAv', 'RTIncorrAv']
        dvs = [col for col in initial_dvs if col in data.columns]
        
        # ===== Parametric Path =====
        # Apply transformations
        logging.info("\n=== Applying Data Transformations ===")
        transformed_data, transforms, boxcox_lambdas = apply_transformations(data, dvs)
        # Save transformation details
        df = pd.DataFrame({
            'DV': list(transforms.keys()),
            'Transform': list(transforms.values()),
            'BoxCox_Lambda': [boxcox_lambdas.get(dv, np.nan) for dv in transforms.keys()]
        })
        df.to_csv(RESULTS_DIR / 'assumptions' / 'data_transformations.csv', index=False)
        
        if len(dvs) >= 2:
            logging.info("\n=== Checking MANOVA Assumptions ===")
            manova_assumptions = check_manova_assumptions(transformed_data, dvs)
            manova_dvs = [
                dv for dv in manova_assumptions['valid_dvs']
                if (manova_assumptions['multivariate_normality'] is not None
                    and manova_assumptions['homogeneity_covariance'] is not None
                    and manova_assumptions['multivariate_normality'] >= 0.05
                    and manova_assumptions['homogeneity_covariance'] >= 0.001)
            ]
        else:
            manova_dvs = []
        
        # Remove multicollinear DVs
        if manova_dvs:
            corr_matrix = transformed_data[manova_dvs].corr().abs()
            upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
            manova_dvs = [col for col in manova_dvs if not any(upper[col] > 0.7)]

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
        art_success = run_nonparametric_analysis(data.copy(), art_dvs)  # Original data
        generate_visualizations(data, art_dvs, 'ART ANOVA')
        
        # Post-hoc analyses
        if manova_success:
            logging.info("\n=== MANOVA Post-hoc Analysis ===")
            # Add MANOVA post-hoc implementation here
            
        if art_success:
            logging.info("\n=== ART ANOVA Post-hoc Analysis ===")
            post_hoc_analysis(data.copy())  # Original data

        # Save comprehensive reports
        save_assumption_reports(manova_assumptions, manova_dvs, art_dvs)
        
    except Exception as e:
        logging.critical(f"Critical pipeline error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
