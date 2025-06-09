import warnings
import shutil
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import sys
import pingouin as pg
from scikit_posthocs import posthoc_conover
from typing import List
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

def initialize_directories():
    """Create required output directories and clean up existing log handlers"""
    # Close existing file handlers
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        if isinstance(handler, logging.FileHandler):
            handler.close()
            root_logger.removeHandler(handler)
    
    # Create directories
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

def _filter_dvs_by_missing(data, dvs, task_level):
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

def _check_rm_assumptions(data, dv):
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
Applied method: *{transforms.get(dv, 'none')}*
*Interpretation:* Transformations adjust data distribution to meet parametric assumptions. Yeo-Johnson preserves sign while handling zero/negative values.

### Assumption Checks
1. *DV Normality (Shapiro-Wilk)*
   - p-value: {assumptions['normality_pval']:.4f}
   - Passed: {'Yes' if assumptions['normality_passed'] else 'No'}
*Interpretation:* Tests if data follows normal distribution (null hypothesis assumes normal distribution). p > 0.05 does not reject null hypothesis."""

    if analysis_type == 'parametric':
        gg_applied = assumptions['epsilon_gg'] < 0.75
        hf_applied = 0.75 <= assumptions['epsilon_gg'] < 1
        correction = 'GG' if gg_applied else 'HF' if hf_applied else 'None'
        
        report_content += f"""
2. *Sphericity Handling*
   - Mauchly's p-value: {assumptions['sphericity_pval']:.4f}
   - Greenhouse-Geisser epsilon: {assumptions['epsilon_gg']:.3f}
   - Correction Applied: {correction}
*Interpretation:* Sphericity requires equal variance differences between conditions. epsilon < 1 indicates violation, requiring correction."""

        if aov is not None:
            if 'F' in aov.columns and 'ddof1' in aov.columns and 'ddof2' in aov.columns:
                f_val = aov['F'].values[0]
                df1 = aov['ddof1'].values[0]
                df2 = aov['ddof2'].values[0]
                eta_sq = (f_val * df1) / (f_val * df1 + df2)
                report_content += f"""
3. *Effect Size*
   - *Partial eta-squared:* {eta_sq:.3f}
*Interpretation:* Proportion of variance explained by factor."""

    eligibility_status = "Eligible for parametric analysis." if assumptions['valid'] else "Ineligible for parametric analysis."
    
    report_content += f"""\n
*Analysis Type:* {analysis_type.title()}
*Eligibility Status:* {eligibility_status}
*Note:* Parametric eligibility requires passed normality (p ≥ 0.05) and sphericity (p ≥ 0.05 or corrected)."""

    report_path = save_dir / f"{dv}_report.md"
    report_path.write_text(report_content.strip(), encoding="utf-8")
    
def _generate_final_report(task_level: str, 
                           nonparametric_path: Path,
                           parametric_path: Path,
                           env_labels: List[str]) -> None:
    """Generates comprehensive final analysis report with both parametric and non-parametric results.
    
    Workflow:
    1. Load analysis results from CSV files
    2. Identify significant/approaching results for both methods
    3. Create method comparison table
    4. Assemble report content with structured sections
    5. Add post-hoc comparisons for both analysis types
    6. Save final markdown report
    """
    
    # Load analysis results from stored CSVs
    df_nonpara = pd.read_csv(nonparametric_path)
    df_para = pd.read_csv(parametric_path)
    
    # Section 1: Significant Findings Identification

    # Filter results using standard significance thresholds
    sig_nonpara = df_nonpara[df_nonpara['p-unc'] < 0.05]
    approach_nonpara = df_nonpara[(df_nonpara['p-unc'] >= 0.05) & (df_nonpara['p-unc'] < 0.1)]
    
    sig_para = df_para[df_para['p-unc'] < 0.05]
    approach_para = df_para[(df_para['p-unc'] >= 0.05) & (df_para['p-unc'] < 0.1)]

    # Section 2: Method Agreement Analysis

    # Compare only variables appearing in significant/approaching results
    relevant_dvs = set(sig_nonpara['DV']).union(
        approach_nonpara['DV'],
        sig_para['DV'],
        approach_para['DV']
    )
    
    comparison = []
    for dv in relevant_dvs:
        nonpara_row = df_nonpara[df_nonpara['DV'] == dv]
        para_row = df_para[df_para['DV'] == dv]
        
        comp_entry = {
            'DV': dv,
            'Nonparametric_p': nonpara_row['p-unc'].values[0] if not nonpara_row.empty else np.nan,
            'Parametric_p': para_row['p-unc'].values[0] if not para_row.empty else np.nan,
            'Agreement': 'Consistent' if (
                nonpara_row['p-unc'].values[0] < 0.05 and 
                para_row['p-unc'].values[0] < 0.05
            ) else 'Discrepancy'
        }
        comparison.append(comp_entry)

    # Section 3: Report Content Assembly

    md_content = f"""## Final Analysis Report ({task_level})

### Environment Definitions
|  Label   |     Description       |
|----------|-----------------------|
| {env_labels[0]:<8} | Base Unity Environment|
| {env_labels[1]:<8} | Nature Environment    |
| {env_labels[2]:<8} | Urban Environment     |

### 1. Significant Nonparametric Results (Friedman Test)
|      Variable      |  p-value  |  W Stat  |  Q Stat  |
|--------------------|-----------|----------|----------|"""

    # Populate nonparametric results table
    for _, row in sig_nonpara.iterrows():
        md_content += f"\n| {row['DV']:<18} | {row['p-unc']:>9.4f} | {row['W']:>8.4f} | {row['Q']:>8.4f} |"

    # Section 4: Approaching Significance

    md_content += "\n\n### 2. Nonparametric Approaching Significance"
    if not approach_nonpara.empty:
        md_content += "\n|      Variable      |   p-value |  W Stat  |  Q Stat  |"
        md_content += "\n|--------------------|-----------|----------|----------|"
        for _, row in approach_nonpara.iterrows():
            md_content += f"\n| {row['DV']:<18} | {row['p-unc']:>9.4f} | {row['W']:>8.4f} | {row['Q']:>8.4f} |"
    else:
        md_content += "\n*No variables approaching significance*"

    # Section 5: Parametric Results

    md_content += "\n\n### 3. Significant Parametric Results (RM ANOVA)"
    md_content += "\n|      Variable      |  p-value  |  F Value  | η²(Generalized) |"
    md_content += "\n|--------------------|-----------|-----------|-----------------|"
    for _, row in sig_para.iterrows():
        md_content += f"\n| {row['DV']:<18} | {row['p-unc']:>9.4f} | {row['F']:>9.4f} | {row['ng2']:>15.4f} |"

    md_content += "\n\n### 4. Parametric Approaching Significance"
    if not approach_para.empty:
        md_content += "\n|      Variable      |  p-value  |  F Value  | η²(Generalized) |"
        md_content += "\n|--------------------|-----------|-----------|-----------------|"
        for _, row in approach_para.iterrows():
            md_content += f"\n| {row['DV']:<18} | {row['p-unc']:>9.4f} | {row['F']:>9.4f} | {row['ng2']:>15.4f} |"
    else:
        md_content += "\n*No variables approaching significance*"

    # Section 6: Method Comparison

    md_content += """\n\n### 5. Method Agreement Analysis
|      Variable       | Nonparametric p | Parametric p |   Agreement   |
|---------------------|-----------------|--------------|---------------|"""
    for row in sorted(comparison, key=lambda x: x['DV']):
        md_content += f"\n| {row['DV']:<19} | {row['Nonparametric_p']:>15.4f} | {row['Parametric_p']:>12.4f} | {row['Agreement']:>13} |"


    # Section 7: Post-Hoc Comparisons

    # Nonparametric Conover post-hocs
    posthoc_path = nonparametric_path.parent / 'posthoc_conover.csv'
    if posthoc_path.exists():
        posthoc_df = pd.read_csv(posthoc_path)
        posthoc_df = posthoc_df[
            (posthoc_df['Env1'] < posthoc_df['Env2']) & 
            (posthoc_df['Env1'] != posthoc_df['Env2'])
        ]
        
        md_content += "\n\n### 6. Nonparametric Post-Hoc Comparisons (Conover's Test)"
        for dv in posthoc_df['DV'].unique():
            md_content += f"\n#### {dv}"
            dv_df = posthoc_df[posthoc_df['DV'] == dv].copy()
            
            # Calculate environment means from analysis data
            analysis_data = pd.read_csv(nonparametric_path.parent.parent / f"{task_level}_data.csv")
            means = analysis_data.groupby('Env')[dv].mean().round(2)
            
            dv_df['Means'] = dv_df.apply(
                lambda row: f"{means[row['Env1']]} vs {means[row['Env2']]}", 
                axis=1
            )
            
            md_content += """
| Comparison of means |  Environments  |  p-value  | Significance |
|---------------------|----------------|-----------|--------------|"""
            for _, row in dv_df.iterrows():
                md_content += f"\n| {row['Means']:<19} | {row['Env1']:7}-{row['Env2']:6} | {row['p_value']:>9.4f} | {row['Significance']:>12} |"

    # Parametric post-hoc comparisons
    para_posthoc_path = parametric_path.parent / 'posthoc_parametric.csv'
    if para_posthoc_path.exists():
        try:
            para_ph = pd.read_csv(para_posthoc_path)
            # Check for required columns and non-empty data
            required_cols = ['Env1', 'Env2', 'p-corr', 'cohen-d', 'DV', 'Significance']
            if not para_ph.empty and all(col in para_ph.columns for col in required_cols):
                md_content += "\n\n### 7. Parametric Post-Hoc Comparisons (Bonferroni-corrected)"
                for dv in para_ph['DV'].unique():
                    md_content += f"\n#### {dv}"
                    dv_df = para_ph[para_ph['DV'] == dv].copy()
                    
                    md_content += """
|  Environments  |   p-value | Cohen's d | Significance |
|----------------|-----------|-----------|--------------|"""
                    for _, row in dv_df.iterrows():
                        md_content += f"\n| {row['Env1']:7}-{row['Env2']:6} | {row['p-corr']:>9.4f} | {row['cohen-d']:>9.4f} | {row['Significance']:>12} |"
            else:
                logging.warning(f"Skipping empty/malformed parametric post-hoc: {para_posthoc_path}")
                
        except Exception as e:
            logging.error(f"Failed to process parametric post-hoc: {str(e)}")

    md_content += """\n
*Statistical Interpretation Note:*
- *Friedman Test:*  
  - W Stat: Kendall’s coefficient (0–1) measuring agreement in ranks. Small: <0.3, Medium: 0.3–0.5, Large: ≥0.5  
  - Q Stat: Friedman’s chi-square statistic testing overall group differences  
- *RM ANOVA:*  
  - F Value: Ratio of between-group to within-group variance  
  - η² (Generalized): Generalized eta squared represents measure of variance explained by environment.
                      Small: 0.01, Medium: 0.06, Large: ≥0.14  
- *Post-Hoc Comparisons:* 
  - p-values: Bonferroni-adjusted significance thresholds (α=0.05)  
  - Cohen’s d: Standardized mean difference. Small: 0.2, Medium: 0.5, Large: ≥0.8  
  - Significance: **: p<0.05 d; ns: p≥0.05 
- *Method Agreement:* Compares parametric and nonparametric p-value consistency"""

    # Final Output
    report_path = nonparametric_path.parent.parent / 'analyses_results.md'
    report_path.write_text(md_content, encoding='utf-8')
    logging.info(f"Generated final report at {report_path}")


    
def _friedman_posthoc(data: pd.DataFrame, task_level: str, dvs: List[str]) -> pd.DataFrame:
    results = []
    for dv in dvs:
        # Get unique environments
        envs = sorted(data['Env'].unique())
        
        # Generate comparison matrix
        ph = posthoc_conover(
            data, 
            val_col=dv, 
            group_col='Env', 
            p_adjust='bonf'
        )
        
        # Convert to long format and filter valid pairs
        ph_long = ph.stack().reset_index()
        ph_long.columns = ['Env1', 'Env2', 'p_value']
        ph_long = ph_long[ph_long['Env1'] != ph_long['Env2']]  # Remove self-comparisons
        
        # Remove duplicate pairs (A vs B == B vs A)
        ph_long = ph_long[ph_long['Env1'] < ph_long['Env2']]
        
        ph_long['DV'] = dv
        ph_long['Task'] = task_level
        ph_long['Significance'] = ph_long['p_value'].apply(
            lambda x: '**' if x < 0.05 else 'ns'
        )
        results.append(ph_long)
    
    return pd.concat(results) if results else pd.DataFrame()

def _parametric_posthoc(data: pd.DataFrame, task_level: str, dvs: List[str]) -> pd.DataFrame:
    """Perform parametric post-hoc tests with Bonferroni correction"""
    results = []
    for dv in dvs:
        try:
            ph = pg.pairwise_tests(
                data=data,
                dv=dv,
                within='Env',
                subject='Subject',
                parametric=True,
                marginal=False,
                alpha=0.05,
                padjust='bonf'
            )
            # Rename columns to match expected format
            ph = ph.rename(columns={
                'A': 'Env1',
                'B': 'Env2',
                'hedges': 'cohen-d'
            })
            # Add significance markers
            ph['Significance'] = ph['p-corr'].apply(
                lambda x: '**' if x < 0.05 else 'ns'
            )
            ph['DV'] = dv
            ph['Task'] = task_level
            results.append(ph)
        except Exception as e:
            logging.error(f"Parametric post-hoc failed for {dv}: {str(e)}")
    return pd.concat(results) if results else pd.DataFrame()

def analyze_task_data(data: pd.DataFrame, 
                     task_level: str, 
                     variables: List[str]) -> List[str]:
    """Core analysis workflow with automated reporting"""
    parametric_dir = RESULTS_DIR / task_level / 'parametric'
    nonparametric_dir = RESULTS_DIR / task_level / 'nonparametric'
    
    filtered_vars = _filter_dvs_by_missing(data, variables, task_level)
    
    parametric_results = []
    nonparametric_results = []
    transformations = {}
    
    # Get actual environment labels from data
    env_labels = sorted(data['Env'].unique())
    
    # Initialize paths
    para_path = parametric_dir / 'analysis_parametric.csv'
    nonpara_path = nonparametric_dir / 'analysis_nonparametric.csv'
    
    # Save task-specific data for reporting
    data.to_csv(RESULTS_DIR / task_level / f"{task_level}_data.csv", index=False)
        
    # Add tracking variables
    parametric_candidates = []
    nonparametric_only = []

    for dv in filtered_vars:
        assumptions = _check_rm_assumptions(data, dv)
        
        # Apply transformation if needed
        if not assumptions['valid'] and not assumptions['normality_passed']:
            pt = PowerTransformer(method='yeo-johnson')
            data[dv] = pt.fit_transform(data[[dv]])
            transformations[dv] = 'yeo-johnson'
            assumptions = _check_rm_assumptions(data, dv)

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

        # Non-parametric analysis path runs always
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

    # Always perform nonparametric post-hoc analysis
    sig_dvs = [row['DV'] for _, row in pd.read_csv(nonpara_path).iterrows() 
              if row['p-unc'] < 0.05]
    if sig_dvs:
        posthoc_df = _friedman_posthoc(data, task_level, sig_dvs)
        posthoc_path = nonparametric_dir / 'posthoc_conover.csv'
        posthoc_df.to_csv(posthoc_path, index=False)
        logging.info(f"Saved post-hoc results to {posthoc_path}")
    
    # Save parametric results
    if parametric_results:
        para_df = pd.concat(parametric_results)
        para_path = parametric_dir / 'analysis_parametric.csv'
        para_df.to_csv(para_path, index=False)
        
        # Perform parametric post-hoc
        sig_para = para_df[para_df['p-unc'] < 0.05]['DV'].tolist()
        if sig_para:
            para_posthoc = _parametric_posthoc(data, task_level, sig_para)
            para_posthoc_path = parametric_dir / 'posthoc_parametric.csv'
            para_posthoc.to_csv(para_posthoc_path, index=False)
            logging.info(f"Saved parametric post-hoc results to {para_posthoc_path}")
    
    # Generate comparative report
    _generate_final_report(
        task_level=task_level,
        nonparametric_path=nonpara_path,
        parametric_path=para_path,
        env_labels=env_labels
    )
    
    # Save task-specific data for reporting
    task_dir = RESULTS_DIR / task_level
    data.to_csv(task_dir / f"{task_level}_data.csv", index=False)
    
    # Add correlation analysis
    _generate_correlation_matrix(data, task_dir)

    return filtered_vars

def _generate_correlation_matrix(task_data: pd.DataFrame, task_dir: Path):
    """Generate and save correlation matrix for task-specific DVs"""
    try:
        # Select only numeric DV columns
        numeric_cols = task_data.select_dtypes(include=np.number).columns.tolist()
        dvs = [col for col in numeric_cols if col not in ['Subject', 'Env', 'N']]
        
        if not dvs:
            logging.warning(f"No valid DVs for correlation matrix in {task_dir.name}")
            return

        # Calculate correlations
        corr_matrix = task_data[dvs].corr(method='pearson')
        
        # Save correlation matrix
        matrix_path = task_dir / 'correlation_matrix.csv'
        corr_matrix.to_csv(matrix_path)
        logging.info(f"Saved correlation matrix to {matrix_path}")

        # Create visualization
        plt.figure(figsize=(12, 10))
        sns.heatmap(
            corr_matrix,
            annot=True,
            fmt=".2f",
            cmap='coolwarm',
            center=0,
            square=True,
            linewidths=0.5,
            cbar_kws={"shrink": 0.8}
        )
        plt.title(f"DV Correlation Matrix ({task_dir.name})", pad=20)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        
        # Save plot
        plot_path = task_dir / 'correlation_matrix.png'
        plt.savefig(plot_path, bbox_inches='tight', dpi=300)
        plt.close()
        logging.info(f"Saved correlation plot to {plot_path}")

    except Exception as e:
        logging.error(f"Correlation matrix generation failed: {str(e)}")

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

        # Move legend to lower right corner
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
        
def generate_motion_artifact_report(data_path: Path, results_dir: Path) -> None:
    """    Generates a markdown report with basic statistics on motion artifacts,
    stratified by baseline (N=0) and experiment (N=2,3)."""
    # Load data
    df = pd.read_csv(data_path)
    logging.info(f"Loaded motion artifact data from {data_path}")
    
    # Define report sections
    report = []
    
    # All data
    overall_stats = df['MotionArt'].describe(percentiles=[.25, .5, .75]).round(4)
    report.append("## Motion Artifact Statistics (All Conditions)")
    report.append("| Statistic |  Value  |")
    report.append("|-----------|---------|")
    for idx, val in overall_stats.items():
        report.append(f"| {idx:9} | {val:7} |")
    logging.info("Computed overall motion artifact statistics")
    
    # Baseline (N=0)
    baseline = df[df['N'] == 0.0]['MotionArt'].describe(percentiles=[.25, .5, .75]).round(4)
    report.append("\n## Baseline (N=0)")
    report.append("| Statistic |  Value  |")
    report.append("|-----------|---------|")
    for idx, val in baseline.items():
        report.append(f"| {idx:9} | {val:7} |")
    logging.info("Computed baseline (N=0) motion artifact statistics")
    
    # Experimental (N=2)
    n2_stats = df[df['N'] == 2.0]['MotionArt'].describe(percentiles=[.25, .5, .75]).round(4)
    report.append("\n## Experimental (N=2)")
    report.append("| Statistic |  Value  |")
    report.append("|-----------|---------|")
    for idx, val in n2_stats.items():
        report.append(f"| {idx:9} | {val:7} |")
    logging.info("Computed N=2 motion artifact statistics")

    # Experimental (N=3)
    n3_stats = df[df['N'] == 3.0]['MotionArt'].describe(percentiles=[.25, .5, .75]).round(4)
    report.append("\n## Experimental (N=3)")
    report.append("| Statistic |  Value  |")
    report.append("|-----------|---------|")
    for idx, val in n3_stats.items():
        report.append(f"| {idx:9} | {val:7} |")
    logging.info("Computed N=3 motion artifact statistics")
    
    # Save report in statistical_analysis folder
    report_path = results_dir / 'motion_artifact_report.md'
    results_dir.mkdir(parents=True, exist_ok=True)
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(report))
    logging.info(f"Saved motion artifact report to {report_path}")

def main():
    """Orchestrates statistical analysis workflow for N-back task data.
    Workflow steps:
    1. Deletes the statistical_analysis folder if it exists
    2. Initializes directory structure
    3. Loads and filters experimental data
    4. Performs parametric/non-parametric analyses
    5. Generates reports and visualizations
    """
    try:
        # Delete existing folder first
        stats_dir = Path("results/statistical_analysis")
        if stats_dir.exists():
            shutil.rmtree(stats_dir)
            logging.shutdown()
            
        # Create new directory structure
        stats_dir.mkdir(parents=True, exist_ok=True)
        
        # 3. Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(stats_dir / 'analysis.log', mode='w'),
                logging.StreamHandler(sys.stdout)
            ]
        )
        logging.info("Logging system initialized after directory cleanup")

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
            task_data_raw = raw_data[raw_data['N'] == task].copy()
            task_data_analysis = task_data_raw.copy()
            current_vars = [v for v in target_vars if v in task_data_analysis.columns]
            filtered_vars = analyze_task_data(task_data_analysis, task_level, current_vars)
            filtered_vars_per_task[task_level] = filtered_vars
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

        # Add motion artifact report
        generate_motion_artifact_report(Path("results/motion_artifacts.csv"), stats_dir)
        
    except Exception as e:
        logging.critical(f"Pipeline failure: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
