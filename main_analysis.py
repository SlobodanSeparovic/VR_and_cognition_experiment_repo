import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
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

def check_repeated_measures_assumptions(data, dvs, manova_mode=False):
    """Check assumptions for MANOVA"""
    assumptions = {
        'normality': {},
        'multicollinearity': None
    }
    
    # Set threshold for MANOVA to 0.7, keep all dependant variables for ANOVA
    threshold = 0.7 if manova_mode else 1.0
    
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

    # 2. Multicollinearity check (drops corelated DVs for MANOVA)
    corr_matrix = data[dvs].corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [col for col in upper.columns if any(upper[col] > threshold)]
    
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
            aligned = df[dv] - df.groupby('Subject')[dv].transform('mean')
            df['ranked'] = rankdata(aligned)
            model = pg.rm_anova(data=df, dv='ranked', 
                               within=['Env','N'], 
                               subject='Subject',
                               detailed=True)
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




def generate_visualizations(data, dvs):
    """Create analysis visualizations with boxplots and overlaid data points."""
    custom_palette = ["#377eb8", "#ff7f00"]
    sns.set_theme(style="whitegrid")
    sns.set_palette(custom_palette)

    for dv in dvs:
        # Boxplot + Stripplot: Distribution by Env and N
        plt.figure(figsize=(8, 6), dpi=300)
        ax = sns.boxplot(
            x='Env', y=dv, hue='N', data=data, palette=custom_palette, showfliers=True, dodge=True
        )
        # Overlay individual data points
        sns.stripplot(
            x='Env', y=dv, hue='N', data=data, 
            palette=custom_palette, dodge=True, jitter=True, 
            marker='o', alpha=0.7, linewidth=0.5, edgecolor='gray', ax=ax
        )
        # Fix legend to avoid double entries
        handles, labels = ax.get_legend_handles_labels()
        n_levels = data['N'].nunique()
        plt.legend(handles[:n_levels], labels[:n_levels], title='N-back', bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
        plt.title(f"Distribution of {dv} by Environment and N-back Level", fontsize=14)
        plt.xlabel("Environment", fontsize=12)
        plt.ylabel(dv, fontsize=12)
        plt.tight_layout()
        plt.savefig(RESULTS_DIR / 'visualizations' / f'{dv}_distribution.png')
        plt.close()

        # Lineplot: Subject trends by Env
        plt.figure(figsize=(8, 6), dpi=300)
        ax = sns.lineplot(x='Env', y=dv, hue='Subject', data=data, legend=False, alpha=0.4, linewidth=1)
        ax.set_title(f"{dv} Subject Trends by Environment", fontsize=14)
        ax.set_xlabel("Environment", fontsize=12)
        ax.set_ylabel(dv, fontsize=12)
        plt.tight_layout()
        plt.savefig(RESULTS_DIR / 'visualizations' / f'{dv}_subject_trends.png')
        plt.close()

        # Pointplot: Interaction effects
        plt.figure(figsize=(8, 6), dpi=300)
        ax = sns.pointplot(
            x='Env', y=dv, hue='N', data=data, 
            errorbar=('ci', 95),
            palette=custom_palette, 
            markers=["o", "s"], linestyles=["-", "--"], dodge=0.2
        )
        ax.set_title(f"{dv} Interaction Effects", fontsize=14)
        ax.set_xlabel("Environment", fontsize=12)
        ax.set_ylabel(dv, fontsize=12)
        plt.legend(title='N-back', bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
        plt.tight_layout()
        plt.savefig(RESULTS_DIR / 'visualizations' / f'{dv}_interaction.png')
        plt.close()
        
def main():
    """Main analysis workflow"""
    try:
        # Create directories
        initialize_directories()
        # Load data     
        data = load_dataset()
        initial_dvs = ['NasaCLI', 'FTCLI_corrected', 'FTPACLI_corrected',
                      'EngI_corrected', 'F1', 'TotalPress', 'IncorrNo', 'CorrRatio', 'RTCorrAv', 'RTIncorrAv']
        dvs = [col for col in initial_dvs if col in data.columns]
        
        # Initial assumption check
        logging.info("\n=== Initial Assumption Check ===")
        assumptions = check_repeated_measures_assumptions(data, dvs)
        
        # Remove multicollinearity handling for all DVs
        final_dvs = [dv for dv in dvs if dv not in assumptions['multicollinearity']]
        
        if len(final_dvs) != len(dvs):
            removed = list(set(dvs) - set(final_dvs))
            logging.warning(f"Removed variables due to multicollinearity: {removed}")
        
        # Apply transformations if needed
        if not all(p >= 0.05 for p in assumptions['normality'].values()):
            logging.info("\n=== Applying Data Transformations ===")
            transformed_data, transforms, boxcox_lambdas = apply_transformations(data, final_dvs)
            # Log transformations with lambda values
            transform_msgs = []
            for dv, method in transforms.items():
                if method == 'boxcox':
                    lambd = boxcox_lambdas.get(dv, 'N/A')
                    transform_msgs.append(f"{dv} ({method}, lambda={lambd:.2f})")
                else:
                    transform_msgs.append(f"{dv} ({method})")
            logging.info("Transformations applied: " + ", ".join(transform_msgs))
            
            # Save lambda values to file
            if boxcox_lambdas:
                pd.DataFrame.from_dict(boxcox_lambdas, orient='index', columns=['lambda']).to_csv(
                    RESULTS_DIR / 'assumptions' / 'boxcox_lambdas.csv')
            
            # Recheck assumptions
            logging.info("\n=== Post-Transformation Assumption Check ===")
            assumptions = check_repeated_measures_assumptions(transformed_data, final_dvs)
        else:
            transformed_data = data.copy()
        
        # Determine analysis path
        parametric_viable = all(p >= 0.05 for p in assumptions['normality'].values())
        
        if parametric_viable:
            logging.info("\n=== Running Parametric Analysis ===")

            # Re-check assumptions with MANOVA-specific threshold
            manova_assumptions = check_repeated_measures_assumptions(transformed_data, final_dvs, manova_mode=True)

            # Apply uniform removal for MANOVA (use final_dvs, not dvs)
            final_dvs = [dv for dv in final_dvs if dv not in manova_assumptions['multicollinearity']]

            if len(final_dvs) != len(dvs):
                removed = list(set(dvs) - set(final_dvs))
                logging.warning(f"Removed due to multicollinearity: {removed}")

            success = run_parametric_manova(transformed_data, final_dvs)
        else:
            # Non-parametric path - keep all DVs regardless of correlations
            logging.warning("Assumptions not met - switching to non-parametric")
            final_dvs = dvs  # Keep all original DVs
            success = run_nonparametric_analysis(transformed_data, final_dvs)
            # Add post-hoc analysis
            if success:
                post_hoc_analysis(transformed_data)
        
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
