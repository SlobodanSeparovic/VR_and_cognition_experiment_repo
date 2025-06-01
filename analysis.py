# analysis.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats
from statsmodels.multivariate.manova import MANOVA
from IPython.display import display  # For notebook display

# Configure paths for notebook compatibility
try:
    BASE_DIR = Path(__file__).parent  # Works in script mode
except NameError:
    BASE_DIR = Path.cwd()  # Fallback for notebook mode

DATA_PATH = BASE_DIR / 'results' / 'final_dataset.csv'
RESULTS_DIR = BASE_DIR / 'results' / 'MANOVA'

def setup_directories():
    """Create required result directories"""
    (RESULTS_DIR / 'assumptions').mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / 'final_results').mkdir(exist_ok=True)
    (RESULTS_DIR / 'visualizations').mkdir(exist_ok=True)

def load_data():
    """Load and clean dataset with validation"""
    print("Loading data...")
    try:
        df = pd.read_csv(DATA_PATH)
    except FileNotFoundError:
        raise FileNotFoundError(f"Data file not found at: {DATA_PATH}")
    
    if 'RTIncorrAv' in df.columns:
        return df.drop(columns=['RTIncorrAv'])
    return df

def check_assumptions(data):
    """Check assumptions and return results + summary"""
    print("\nChecking assumptions...")
    
    # 1. Normality checks
    dvs = ['NasaCLI', 'FTCLI_corrected', 'FTPACLI_corrected',
           'EngI_corrected', 'F1', 'TotalPress', 'CorrRatio', 'RTCorrAv']
    
    # Perform tests
    normality_results = []
    for dv in dvs:
        for env in data['Env'].unique():
            for n_level in [2.0, 3.0]:
                subset = data[(data['Env'] == env) & (data['N'] == n_level)][dv]
                stat, p = stats.shapiro(subset)
                normality_results.append({
                    'DV': dv, 'Env': env, 'N': n_level,
                    'W-stat': stat, 'p-value': p
                })

    # Save and analyze
    normality_df = pd.DataFrame(normality_results)
    normality_df.to_csv(RESULTS_DIR / 'assumptions' / 'normality_tests.csv', index=False)
    
    # 2. Homogeneity of variances
    levene_results = []
    for dv in dvs:
        groups = data.groupby(['Env', 'N'])[dv].apply(list)
        stat, p = stats.levene(*groups)
        levene_results.append({'DV': dv, 'F': stat, 'p-value': p})
    
    levene_df = pd.DataFrame(levene_results)
    levene_df.to_csv(RESULTS_DIR / 'assumptions' / 'homogeneity_tests.csv', index=False)

    # 3. Multicollinearity check
    corr_matrix = data[dvs].corr()
    plt.figure(figsize=(12, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title("Correlation Matrix of Dependent Variables")
    plt.savefig(RESULTS_DIR / 'assumptions' / 'correlation_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Generate feedback
    norm_violations = normality_df[normality_df['p-value'] < 0.05]
    levene_violations = levene_df[levene_df['p-value'] < 0.05]
    high_corr = corr_matrix.abs().stack().reset_index()
    high_corr = high_corr[(high_corr['level_0'] != high_corr['level_1']) & (high_corr[0] > 0.7)]

    feedback = {
        'normality': norm_violations,
        'homogeneity': levene_violations,
        'correlation': high_corr,
        'summary': [
            f"Normality violations: {len(norm_violations)}",
            f"Homogeneity violations: {len(levene_violations)}",
            f"High correlations (>0.7): {len(high_corr)}"
        ]
    }
    
    return feedback

def transform_variables(data):
    """Apply transformations and return cleaned data"""
    print("\nApplying transformations...")
    
    # Remove correlated variables
    df_clean = data.drop(columns=['CorrRatio', 'FTCLI_corrected'])
    
    # Apply transformations
    df_clean['log_FTPACLI'] = np.log(df_clean['FTPACLI_corrected'] + 1)
    df_clean['rank_EngI'] = df_clean['EngI_corrected'].rank()
    df_clean['log_TotalPress'] = np.log(df_clean['TotalPress'] + 1)
    df_clean['asin_F1'] = np.arcsin(np.sqrt(df_clean['F1']))
    
    return df_clean

def run_manova(data):
    """Run MANOVA and return results"""
    print("\nRunning MANOVA...")
    
    formula = '''NasaCLI + log_FTPACLI + rank_EngI + 
                asin_F1 + log_TotalPress + RTCorrAv ~ Env * N'''
    
    manova = MANOVA.from_formula(formula, data=data)
    results = manova.mv_test()
    
    # Save results
    with open(RESULTS_DIR / 'final_results' / 'manova_results.txt', 'w') as f:
        f.write(results.summary().as_text())
    
    return results

def posthoc_analysis(data):
    """Run posthoc tests and return results"""
    print("\nRunning post-hoc tests...")
    
    tests = []
    for dv in ['NasaCLI', 'asin_F1', 'log_TotalPress']:
        # Environment effects
        for n_level in [2.0, 3.0]:
            groups = [
                data[(data['Env'] == 'Control') & (data['N'] == n_level)][dv],
                data[(data['Env'] == 'Nature') & (data['N'] == n_level)][dv],
                data[(data['Env'] == 'Urban') & (data['N'] == n_level)][dv]
            ]
            h_stat, p_val = stats.kruskal(*groups)
            tests.append({'Test': f'{dv} Env@N{n_level}', 'p-value': p_val})
        
        # N-back effects
        for env in ['Control', 'Nature', 'Urban']:
            group1 = data[(data['Env'] == env) & (data['N'] == 2.0)][dv]
            group2 = data[(data['Env'] == env) & (data['N'] == 3.0)][dv]
            u_stat, p_val = stats.mannwhitneyu(group1, group2)
            tests.append({'Test': f'{dv} N@{env}', 'p-value': p_val})
    
    tests_df = pd.DataFrame(tests)
    tests_df.to_csv(RESULTS_DIR / 'final_results' / 'posthoc_results.csv', index=False)
    return tests_df

def generate_visualizations(data):
    """Generate and save visualizations"""
    print("\nCreating visualizations...")
    
    # F1 score plot
    plt.figure(figsize=(10, 6))
    sns.barplot(
        x='Env', 
        y='asin_F1', 
        hue='N',
        data=data,
        palette={2.0: "#1f77b4", 3.0: "#ff7f0e"},
        errorbar=('ci', 95)
    )
    plt.title("Task Accuracy (F1) by Environment and N-back Level")
    plt.ylabel("Transformed F1 Score")
    plt.savefig(RESULTS_DIR / 'visualizations' / 'f1_scores.png', dpi=300)
    plt.close()
    
    return "Visualizations saved to results folder"

# For notebook execution
if __name__ != "__main__":
    # Setup when imported
    setup_directories()