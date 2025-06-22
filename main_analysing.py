import warnings
import shutil
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import pingouin as pg
from scikit_posthocs import posthoc_conover
from typing import List
from pathlib import Path
from sklearn.preprocessing import PowerTransformer
import scipy.stats

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
    """Create required output directories"""   
    for task in ['2back', '3back']:
        (RESULTS_DIR / task / 'parametric').mkdir(parents=True, exist_ok=True)
        (RESULTS_DIR / task / 'nonparametric').mkdir(parents=True, exist_ok=True)

def load_dataset():
    """Load and validate input dataset from CSV"""
    try:
        df = pd.read_csv(DATA_PATH)
        print(f"Info: Dataset loaded from {DATA_PATH}")
        return df
    except FileNotFoundError:
        print("Critical warning: Missing input data file - aborting analysis")
        sys.exit(1)

def _check_rm_assumptions(data, dv):
    """Check assumptions for parametric analysis with environment-specific normality"""
    env_normality = {}
    all_passed = True
    
    for env in data['Env'].unique():
        env_data = data[data['Env'] == env][dv]
        # Detect zero-range data (all values identical)
        zero_range = (env_data.nunique() == 1)
        norm = pg.normality(env_data)
        env_passed = norm['pval'].values[0] >= 0.05
        env_normality[env] = {
            'pval': norm['pval'].values[0],
            'passed': env_passed,
            'zero_range': zero_range  # New flag
        }
        if not env_passed:
            all_passed = False
    
    sph = pg.sphericity(data, dv=dv, subject='Subject', within='Env')
    
    return {
        'env_normality': env_normality,
        'all_envs_normal': all_passed,
        'sphericity_pval': sph[-1],
        'epsilon_gg': pg.epsilon(data, dv=dv, correction='gg'),
        'valid': all_passed and (sph[-1] >= 0.05)
    }

def generate_report(initial_assumptions, final_assumptions, transforms, dv, task_level, save_dir, analysis_type='parametric', aov=None, dv_descriptive=None):
    """Create type-specific analysis reports with pre/post transformation comparison"""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Determine transformation status for reporting
    transformed = dv in transforms
    transformation_status = "*Pre/Post Transformation Comparison*" if transformed else "*No Transformation Applied*"
    
    report_content = f"""
## Analysis Report: {dv} ({task_level})
{transformation_status}

### Data Transformations
Applied method: *{transforms.get(dv, 'none')}*
*Interpretation:* Transformations adjust data distribution to meet parametric assumptions. Yeo-Johnson preserves sign while handling zero/negative values.

### Pre-Transformation Assumptions
1. *DV Normality (Shapiro-Wilk) by Environment*
   | Environment | p-value  | Passed (p≥0.05)  | Note          |
   |-------------|----------|------------------|---------------|"""
    
    for env, results in initial_assumptions['env_normality'].items():
        note = "Zero range" if results['zero_range'] else ""
        report_content += f"\n   | {env:<11} | {results['pval']:>8.4f} | {'Yes' if results['passed'] else 'No':>16} | {note:<13} |"
    
    report_content += f"""
*Overall normality:* {'All environments passed' if initial_assumptions['all_envs_normal'] else 'At least one environment failed normality'}"""

    # post-transformation section if applied
    if transformed:
        report_content += """
        
### Post-Transformation Assumptions
1. *DV Normality (Shapiro-Wilk) by Environment*
   | Environment | p-value  | Passed (p≥0.05)  | Note          |
   |-------------|----------|------------------|---------------|"""
        
        for env, results in final_assumptions['env_normality'].items():
            note = "Zero range" if results['zero_range'] else ""
            report_content += f"\n   | {env:<11} | {results['pval']:>8.4f} | {'Yes' if results['passed'] else 'No':>16} | {note:<13} |"
        
        report_content += f"""
*Overall normality:* {'All environments passed' if final_assumptions['all_envs_normal'] else 'At least one environment failed normality'}"""
    
    # explanatory note for zero-range cases
    report_content += """
\n**Note:** Environments marked as 'Zero range' have all values identical. 
Normality tests (Shapiro-Wilk) are unreliable in these cases."""

    if analysis_type == 'parametric':
        correction = 'GG' if final_assumptions['epsilon_gg'] < 1 else 'None'
        
        report_content += f"""
        
2. *Sphericity Handling*
   - Mauchly's p-value: {final_assumptions['sphericity_pval']:.4f}
   - Greenhouse-Geisser epsilon: {final_assumptions['epsilon_gg']:.3f}
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
    
    if dv_descriptive:
        report_content += "\n\n### Descriptive Statistics (Original Scale)"
        report_content += "\n| Environment |   Mean   |      95% CI      |"
        report_content += "\n|-------------|----------|------------------|"
        for env, stats in dv_descriptive.items():
            ci_str = f"[{stats['ci_low']:.2f}, {stats['ci_high']:.2f}]"
            report_content += f"\n| {env:<11} | {stats['mean']:<8.2f} | {ci_str:<16} |"
            
    eligibility_status = "Eligible for parametric analysis." if final_assumptions['valid'] else "Ineligible for parametric analysis."        
    report_content += f"""\n
*Eligibility Status:* {eligibility_status}
*Analysis Type:* {analysis_type.title()}{' and nonparametric' if analysis_type == 'parametric' else ''}
*Note:* Parametric eligibility requires passed normality (p ≥ 0.05) in all environments and sphericity (p ≥ 0.05 or corrected)."""

    report_path = save_dir / f"{dv}_report.md"
    report_path.write_text(report_content.strip(), encoding="utf-8")

def _generate_final_report(task_level: str, 
                           nonparametric_path: Path,
                           parametric_path: Path,
                           env_labels: List[str],
                           descriptive_stats: dict) -> None:
    """Generates comprehensive final analysis report with both parametric and non-parametric results."""
    df_nonpara = pd.read_csv(nonparametric_path)
    df_para = pd.read_csv(parametric_path)
    
    sig_nonpara = df_nonpara[df_nonpara['p-unc'] < 0.05]
    approach_nonpara = df_nonpara[(df_nonpara['p-unc'] >= 0.05) & (df_nonpara['p-unc'] < 0.1)]
    
    sig_para = df_para[df_para['p-unc'] < 0.05]
    approach_para = df_para[(df_para['p-unc'] >= 0.05) & (df_para['p-unc'] < 0.1)]

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

    for _, row in sig_nonpara.iterrows():
        md_content += f"\n| {row['DV']:<18} | {row['p-unc']:>9.4f} | {row['W']:>8.4f} | {row['Q']:>8.4f} |"

    md_content += "\n\n### 2. Nonparametric Approaching Significance"
    if not approach_nonpara.empty:
        md_content += "\n|      Variable      |   p-value |  W Stat  |  Q Stat  |"
        md_content += "\n|--------------------|-----------|----------|----------|"
        for _, row in approach_nonpara.iterrows():
            md_content += f"\n| {row['DV']:<18} | {row['p-unc']:>9.4f} | {row['W']:>8.4f} | {row['Q']:>8.4f} |"
    else:
        md_content += "\n*No variables approaching significance*"

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

    md_content += """\n\n### 5. Method Agreement Analysis
|      Variable       | Nonparametric p | Parametric p |   Agreement   |
|---------------------|-----------------|--------------|---------------|"""
    for row in sorted(comparison, key=lambda x: x['DV']):
        md_content += f"\n| {row['DV']:<19} | {row['Nonparametric_p']:>15.4f} | {row['Parametric_p']:>12.4f} | {row['Agreement']:>13} |"

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
            
            dv_stats = descriptive_stats.get(dv, {})
            
            dv_df['Means'] = dv_df.apply(
                lambda row: f"{dv_stats.get(row['Env1'], {}).get('mean', 'NA'):.2f} vs {dv_stats.get(row['Env2'], {}).get('mean', 'NA'):.2f}", 
                axis=1
            )
            
            md_content += """
|  Environments  | Comparison of means |  p-value  | Significance |
|----------------|---------------------|-----------|--------------|"""
            for _, row in dv_df.iterrows():
                md_content += f"\n| {row['Env1']:7}-{row['Env2']:6} | {row['Means']:<19} | {row['p_value']:>9.4f} | {row['Significance']:>12} |"

    para_posthoc_path = parametric_path.parent / 'posthoc_parametric.csv'
    if para_posthoc_path.exists():
        try:
            para_ph = pd.read_csv(para_posthoc_path)
            required_cols = ['Env1', 'Env2', 'p-corr', 'cohen-d', 'DV', 'Significance']
            if not para_ph.empty and all(col in para_ph.columns for col in required_cols):
                md_content += "\n\n### 7. Parametric Post-Hoc Comparisons (Bonferroni-corrected)"
                for dv in para_ph['DV'].unique():
                    md_content += f"\n#### {dv}"
                    dv_df = para_ph[para_ph['DV'] == dv].copy()
                    
                    dv_stats = descriptive_stats.get(dv, {})
                    dv_df['Means'] = dv_df.apply(
                        lambda row: f"{dv_stats.get(row['Env1'], {}).get('mean', 'NA'):.2f} vs {dv_stats.get(row['Env2'], {}).get('mean', 'NA'):.2f}", 
                        axis=1
                    )
                    
                    md_content += """
|  Environments  | Comparison of means |   p-value | Cohen's d | Significance |
|----------------|---------------------|-----------|-----------|--------------|"""
                    for _, row in dv_df.iterrows():
                        md_content += f"\n| {row['Env1']:7}-{row['Env2']:6} | {row['Means']:<19} | {row['p-corr']:>9.4f} | {row['cohen-d']:>9.4f} | {row['Significance']:>12} |"
            else:
                print(f"Warning: Skipping empty/malformed parametric post-hoc: {para_posthoc_path}")
                
        except Exception as e:
            print(f"Error: Failed to process parametric post-hoc: {str(e)}")

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

    report_path = nonparametric_path.parent.parent / 'analyses_results.md'
    report_path.write_text(md_content, encoding='utf-8')
    print(f"Info: Generated final report at {report_path}")

def _friedman_posthoc(data: pd.DataFrame, task_level: str, dvs: List[str]) -> pd.DataFrame:
    """Perform non-parametric post-hoc pairwise Conover tests with Bonferroni correction"""
    results = []
    for dv in dvs:
        envs = sorted(data['Env'].unique())
        ph = posthoc_conover(
            data, 
            val_col=dv, 
            group_col='Env', 
            p_adjust='bonf'
        )
        ph_long = ph.stack().reset_index()
        ph_long.columns = ['Env1', 'Env2', 'p_value']
        ph_long = ph_long[ph_long['Env1'] != ph_long['Env2']]
        ph_long = ph_long[ph_long['Env1'] < ph_long['Env2']]
        ph_long['DV'] = dv
        ph_long['Task'] = task_level
        ph_long['Significance'] = ph_long['p_value'].apply(
            lambda x: '**' if x < 0.05 else 'ns'
        )
        results.append(ph_long)
    return pd.concat(results) if results else pd.DataFrame()

def _parametric_posthoc(data: pd.DataFrame, task_level: str, dvs: List[str]) -> pd.DataFrame:
    """Perform parametric post-hoc pairwise t-tests with Bonferroni correction"""
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
            ph = ph.rename(columns={
                'A': 'Env1',
                'B': 'Env2',
                'hedges': 'cohen-d'
            })
            ph['Significance'] = ph['p-corr'].apply(
                lambda x: '**' if x < 0.05 else 'ns'
            )
            ph['DV'] = dv
            ph['Task'] = task_level
            results.append(ph)
        except Exception as e:
            print(f"Error: Parametric post-hoc failed for {dv}: {str(e)}")
    return pd.concat(results) if results else pd.DataFrame()

def analyze_task_data(data: pd.DataFrame, task_level: str, variables: List[str]) -> List[str]:
    """Core analysis workflow with automated reporting and descriptive statistics"""
    parametric_dir = RESULTS_DIR / task_level / 'parametric'
    nonparametric_dir = RESULTS_DIR / task_level / 'nonparametric'
    
    parametric_results = []
    nonparametric_results = []
    transformations = {}
    transformers = {}  # Store transformer objects for back-transformation
    descriptive_stats = {}  # Store descriptive statistics for reporting
    assumption_history = {}  # Track assumption states before/after transformations
    
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

    for dv in variables:
        # Store original data for back-transformation
        original_dv = data[dv].copy()
        
        # Capture pre-transformation assumption state
        pre_assumptions = _check_rm_assumptions(data, dv)
        assumption_history[dv] = {'pre': pre_assumptions}
        
        # Apply transformation if needed based on pre-assumptions
        if not pre_assumptions['valid'] and not pre_assumptions['all_envs_normal']:
            pt = PowerTransformer(method='yeo-johnson')
            data[[dv]] = pt.fit_transform(data[[dv]])
            transformations[dv] = 'yeo-johnson'
            transformers[dv] = pt
            # Capture post-transformation assumption state
            post_assumptions = _check_rm_assumptions(data, dv)
            assumption_history[dv]['post'] = post_assumptions
        else:
            post_assumptions = pre_assumptions

        # Track analysis eligibility using final assumption state
        if post_assumptions['valid']:
            parametric_candidates.append(dv)
        else:
            nonparametric_only.append(dv)

        # Compute descriptive statistics (on original scale)
        dv_descriptive = {}
        for env in env_labels:
            if dv in transformers:
                # Back-transform stats for reporting
                trans_df = data.loc[data['Env'] == env, [dv]]
                mean_trans = np.mean(trans_df[dv])
                sem_trans = scipy.stats.sem(trans_df[dv])
                n = len(trans_df)
                if n > 1:
                    ci_low_trans, ci_high_trans = scipy.stats.t.interval(0.95, n-1, loc=mean_trans, scale=sem_trans)
                else:
                    ci_low_trans, ci_high_trans = mean_trans, mean_trans
                
                transformer = transformers[dv]
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=UserWarning)
                    mean_orig = transformer.inverse_transform(pd.DataFrame({dv: [mean_trans]}))[0][0]
                    ci_low_orig = transformer.inverse_transform(pd.DataFrame({dv: [ci_low_trans]}))[0][0]
                    ci_high_orig = transformer.inverse_transform(pd.DataFrame({dv: [ci_high_trans]}))[0][0]
            else:
                env_data = data[data['Env'] == env][dv].dropna().values
                mean_orig = np.mean(env_data)
                sem_orig = scipy.stats.sem(env_data)
                n = len(env_data)
                if n > 1:
                    ci_low_orig, ci_high_orig = scipy.stats.t.interval(0.95, n-1, loc=mean_orig, scale=sem_orig)
                else:
                    ci_low_orig, ci_high_orig = mean_orig, mean_orig
            
            dv_descriptive[env] = {
                'mean': mean_orig,
                'ci_low': ci_low_orig,
                'ci_high': ci_high_orig
            }
        
        descriptive_stats[dv] = dv_descriptive

        # Parametric analysis path
        if post_assumptions['valid']:
            try:
                aov = pg.rm_anova(
                    data=data,
                    dv=dv,
                    within='Env',
                    subject='Subject',
                    correction='auto'
                ).assign(DV=dv)
                
                parametric_cols = ['DV'] + \
                                [c for c in aov.columns if c.startswith('p-')] + \
                                [c for c in aov.columns if c not in ['DV'] and not c.startswith('p-')]
                aov = aov[parametric_cols]
                
                parametric_results.append(aov)
                generate_report(
                    initial_assumptions=assumption_history[dv]['pre'],
                    final_assumptions=post_assumptions,
                    transforms=transformations,
                    dv=dv,
                    task_level=task_level,
                    save_dir=parametric_dir,
                    analysis_type='parametric',
                    aov=aov,
                    dv_descriptive=dv_descriptive
                )
            except Exception as e:
                print(f"Error: Parametric analysis failed for {dv}: {str(e)}")
        
        # Non-parametric analysis path runs always
        try:
            friedman = pg.friedman(
                data=data,
                dv=dv,
                within='Env',
                subject='Subject'
            ).assign(DV=dv)
            
            nonparametric_cols = ['DV'] + \
                               [c for c in friedman.columns if c.startswith('p-')] + \
                               [c for c in friedman.columns if c not in ['DV'] and not c.startswith('p-')]
            friedman = friedman[nonparametric_cols]
            
            nonparametric_results.append(friedman)
            
            if not post_assumptions['valid']:
                generate_report(
                    initial_assumptions=assumption_history[dv]['pre'],
                    final_assumptions=post_assumptions,
                    transforms=transformations,
                    dv=dv,
                    task_level=task_level,
                    save_dir=nonparametric_dir,
                    analysis_type='nonparametric',
                    dv_descriptive=dv_descriptive
                )
        except Exception as e:
            print(f"Error: Non-parametric analysis failed for {dv}: {str(e)}")
    
    # Save results with proper column order
    if parametric_results:
        pd.concat(parametric_results).to_csv(parametric_dir/'analysis_parametric.csv', index=False)
    
    if nonparametric_results:
        pd.concat(nonparametric_results).to_csv(nonparametric_dir/'analysis_nonparametric.csv', index=False)
    
    # Print analysis eligibility
    if parametric_candidates:
        print(f"Info: {task_level} variables eligible for both analyses: {', '.join(parametric_candidates)}")
    
    if nonparametric_only:
        print(f"Info: {task_level} variables requiring non-parametric analysis: {', '.join(nonparametric_only)}")

    # Perform nonparametric post-hoc analysis
    sig_dvs = [row['DV'] for _, row in pd.read_csv(nonpara_path).iterrows() 
              if row['p-unc'] < 0.05] if nonpara_path.exists() else []
    if sig_dvs:
        posthoc_df = _friedman_posthoc(data, task_level, sig_dvs)
        posthoc_path = nonparametric_dir / 'posthoc_conover.csv'
        posthoc_df.to_csv(posthoc_path, index=False)
        print(f"Info: Saved post-hoc results to {posthoc_path}")
    
    # Save parametric results and post-hoc
    if parametric_results:
        para_df = pd.concat(parametric_results)
        para_path = parametric_dir / 'analysis_parametric.csv'
        para_df.to_csv(para_path, index=False)
        
        sig_para = para_df[para_df['p-unc'] < 0.05]['DV'].tolist()
        if sig_para:
            para_posthoc = _parametric_posthoc(data, task_level, sig_para)
            para_posthoc_path = parametric_dir / 'posthoc_parametric.csv'
            para_posthoc.to_csv(para_posthoc_path, index=False)
            print(f"Info: Saved parametric post-hoc results to {para_posthoc_path}")
    
    # Generate comparative report with descriptive stats
    _generate_final_report(
        task_level=task_level, 
        nonparametric_path=nonpara_path, 
        parametric_path=para_path, 
        env_labels=env_labels,
        descriptive_stats=descriptive_stats
    )
    
    # Save task-specific data
    task_dir = RESULTS_DIR / task_level
    data.to_csv(task_dir / f"{task_level}_data.csv", index=False)
    
    # Add correlation analysis
    _generate_correlation_matrix(data, task_dir)

    return variables

def _generate_correlation_matrix(task_data: pd.DataFrame, task_dir: Path):
    """Generate and save correlation matrix for task-specific DVs"""
    try:
        numeric_cols = task_data.select_dtypes(include=np.number).columns.tolist()
        dvs = [col for col in numeric_cols if col not in ['Subject', 'Env', 'N']]
        
        if not dvs:
            print(f"Warning: No valid DVs for correlation matrix in {task_dir.name}")
            return
        
        corr_matrix = task_data[dvs].corr(method='pearson')
        matrix_path = task_dir / 'correlation_matrix.csv'
        corr_matrix.to_csv(matrix_path)
        print(f"Info: Saved correlation matrix to {matrix_path}")
        
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
        
        plot_path = task_dir / 'correlation_matrix.png'
        plt.savefig(plot_path, bbox_inches='tight', dpi=300)
        plt.close()
        print(f"Info: Saved correlation plot to {plot_path}")

    except Exception as e:
        print(f"Error: Correlation matrix generation failed: {str(e)}")

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
    """Generates a markdown report with basic statistics on motion artifacts"""
    df = pd.read_csv(data_path)
    print(f"Info: Loaded motion artifact data from {data_path}")
    
    report = []
    
    overall_stats = df['MotionArt'].describe(percentiles=[.25, .5, .75]).round(4)
    report.append("## Motion Artifact Statistics (All Conditions)")
    report.append("| Statistic |  Value  |")
    report.append("|-----------|---------|")
    for idx, val in overall_stats.items():
        report.append(f"| {idx:9} | {val:7} |")
    print("Info: computed overall motion artifact statistics")
    
    baseline = df[df['N'] == 0.0]['MotionArt'].describe(percentiles=[.25, .5, .75]).round(4)
    report.append("\n## Baseline (N=0)")
    report.append("| Statistic |  Value  |")
    report.append("|-----------|---------|")
    for idx, val in baseline.items():
        report.append(f"| {idx:9} | {val:7} |")
    print("Info: computed baseline (N=0) motion artifact statistics")
    
    n2_stats = df[df['N'] == 2.0]['MotionArt'].describe(percentiles=[.25, .5, .75]).round(4)
    report.append("\n## Experimental (N=2)")
    report.append("| Statistic |  Value  |")
    report.append("|-----------|---------|")
    for idx, val in n2_stats.items():
        report.append(f"| {idx:9} | {val:7} |")
    print("Info: computed N=2 motion artifact statistics")

    n3_stats = df[df['N'] == 3.0]['MotionArt'].describe(percentiles=[.25, .5, .75]).round(4)
    report.append("\n## Experimental (N=3)")
    report.append("| Statistic |  Value  |")
    report.append("|-----------|---------|")
    for idx, val in n3_stats.items():
        report.append(f"| {idx:9} | {val:7} |")
    print("Info: computed N=3 motion artifact statistics")
    
    report_path = results_dir / 'motion_artifact_report.md'
    results_dir.mkdir(parents=True, exist_ok=True)
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(report))
    print(f"Info: Saved motion artifact report to {report_path}")

def main():
    """Orchestrates statistical analysis workflow for N-back task data"""
    try:
        stats_dir = Path("results/statistical_analysis")
        if stats_dir.exists():
            shutil.rmtree(stats_dir)
        stats_dir.mkdir(parents=True, exist_ok=True)

        initialize_directories()
        raw_data = load_dataset()
        target_vars = [
            'NasaCLI', 'FTCLI_corrected', 'FTPACLI_corrected',
            'EngI_corrected', 'F1', 'TotalPress', 'IncorrNo', 
            'CorrRatio', 'RTCorrAv', 'RTIncorrAv'
        ]
        
        all_viz_data = []
        available_vars_per_task = {}

        for task in [2, 3]:
            task_level = f"{task}back"
            task_data_raw = raw_data[raw_data['N'] == task].copy()
            task_data_analysis = task_data_raw.copy()
            current_vars = [v for v in target_vars if v in task_data_analysis.columns]
            available_vars = analyze_task_data(task_data_analysis, task_level, current_vars)
            available_vars_per_task[task_level] = available_vars
            viz_data = task_data_raw[['Subject', 'Env'] + available_vars].copy()
            viz_data['Task'] = task_level
            all_viz_data.append(viz_data)
            print(f"Info: {task_level} variables: {', '.join(available_vars)}")

        if all_viz_data:
            common_dvs = set.intersection(*[set(vars) for vars in available_vars_per_task.values()])
            if common_dvs:
                print(f"Info: Common variables in 2-back and 3-back conditions: {', '.join(common_dvs)}")
                combined_data = pd.concat(all_viz_data)
                generate_combined_visualizations(combined_data, list(common_dvs))
            else:
                print("Warning: No common variables across tasks - generating task-specific plots")
                for task_level, vars in available_vars_per_task.items():
                    task_data = [d for d in all_viz_data if d['Task'].iloc[0] == task_level][0]
                    generate_combined_visualizations(task_data, vars)

        generate_motion_artifact_report(Path("results/motion_artifacts.csv"), stats_dir)
        
    except Exception as e:
        print(f"Critical warning: Pipeline failure: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()