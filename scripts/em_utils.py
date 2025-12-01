import matplotlib.pyplot as plt
import os
import numpy as np
from scipy import stats
import polars as pl
from sklearn.metrics import roc_curve, auc
import matplotlib.image as mpimg
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

def compute_QQ_validate_distribution_robust(out_df, variable = "s_l", addition = 1, save_figure_path = "../plots/test.distribution.png"):
    """More robust distribution testing with additional metrics"""
    
    sources = [s for s in out_df['source'].unique().to_list() if s != 'non_determined']
    distributions = [('Gamma', stats.gamma), ('Normal', stats.norm), ('Student-t', stats.t)]
    
    fig, axes = plt.subplots(len(sources), 3, figsize=(8, 2*len(sources)))
    if len(sources) == 1:
        axes = axes.reshape(1, -1)
    
    results = {}
    
    for i, source in enumerate(sources):
        log_data = np.log(out_df.filter(pl.col('source') == source)[variable].to_numpy() + addition)
        results[source] = {}
        
        print(f"\n{source.upper()} (n={len(log_data)}):")
        print("-" * 40)
        
        for j, (name, dist) in enumerate(distributions):
            # Fit distribution
            if name == 'Gamma':
                params = dist.fit(log_data, floc=0)
            else:
                params = dist.fit(log_data)
            
            # Q-Q plot
            stats.probplot(log_data, dist=dist, sparams=params, plot=axes[i, j])
            
            # Multiple goodness-of-fit measures
            # 1. R-squared from Q-Q plot
            theoretical = axes[i, j].get_lines()[1].get_xdata()
            empirical = axes[i, j].get_lines()[1].get_ydata()
            r2 = np.corrcoef(theoretical, empirical)[0, 1]**2
            
            # 2. Anderson-Darling test (more sensitive to tails)
            try:
                if name == 'Normal':
                    ad_stat, ad_crit, ad_sig = stats.anderson(log_data, dist='norm')
                    ad_p = 1 - ad_sig[np.searchsorted(ad_crit, ad_stat)] / 100 if ad_stat < ad_crit[-1] else 0.001
                else:
                    ad_p = None
            except:
                ad_p = None
            
            # 3. Shapiro-Wilk for normality (if normal)
            if name == 'Normal' and len(log_data) <= 5000:
                sw_stat, sw_p = stats.shapiro(log_data)
            else:
                sw_p = None
            
            # 4. KS test (but interpret cautiously)
            ks_stat, ks_p = stats.kstest(log_data, lambda x: dist.cdf(x, *params))
            
            # 5. Visual deviation measure
            qq_deviation = np.mean(np.abs(empirical - theoretical))
            
            axes[i, j].set_title(f'{source} - {name}\nR²={r2:.3f}, QQ-dev={qq_deviation:.3f}\nKS p={ks_p:.3f}')
            axes[i, j].grid(True, alpha=0.3)
            
            # Store comprehensive results
            results[source][name] = {
                'r2': r2, 
                'ks_p': ks_p, 
                'qq_deviation': qq_deviation,
                'ad_p': ad_p,
                'sw_p': sw_p,
                'params': params,
                'n_samples': len(log_data)
            }
            
            # Print detailed results
            print(f"{name:12} | R²={r2:.3f} | QQ-dev={qq_deviation:.3f} | KS p={ks_p:.3f}", end="")
            if ad_p is not None:
                print(f" | AD p={ad_p:.3f}", end="")
            if sw_p is not None:
                print(f" | SW p={sw_p:.3f}", end="")
            
            # Overall assessment
            good_indicators = sum([
                r2 > 0.95,
                qq_deviation < 0.5,
                ks_p > 0.05,
                ad_p > 0.05 if ad_p else False,
                sw_p > 0.05 if sw_p else False
            ])
            
            if good_indicators >= 3:
                print(" | ✓ GOOD")
            elif good_indicators >= 2:
                print(" | ~ FAIR") 
            else:
                print(" | ✗ POOR")
    
    plt.tight_layout()
    plt.savefig(save_figure_path)
    plt.show()
    
    return results

def heatmap_plot_znseq(data: pl.DataFrame, 
                 columns: list[str], 
                 no_normalize_columns: list[str], 
                 save_heatmap_path: str, file_title: str, motif_logo_i: str, rbp_df: pl.DataFrame, sample_index):

    # Add professional label mapping
    label_mapping = {
        'TSS_proximity': 'TSS Proximity',
        'score_phastcons100': 'PhastCons100way score',
        'score_phylop100': 'PhyloP100way score', 
        'PhastCons100_percent': 'PhastCons100way region score ',
        'inner_mean_logPWM': 'PWM score: inner region',
        'outer_mean_logPWM': 'PWM score: outer region',
        'GC_inner_pct': "GC content: inner region",
        'GC_outer_pct': 'GC content: outer region',
        'AS_native_mean_inner': 'Native AS score',
        'AS_fixed_mean_inner': 'Fixed AS score',
        'Znseq_logsignal': 'Zn-seq signal',
        'prior_p': 'Prior',
        'posterior_r': 'Posterior',
        'source': 'CLIP label'
    }

    special_transforms = {'s_l': lambda x: np.log(x + 1)}

    data = data.select(columns).to_pandas()
    zscore_to_01 = lambda x: (stats.zscore(x) - stats.zscore(x).min()) / (stats.zscore(x).max() - stats.zscore(x).min())

    for col in columns[:-1]:
        if col in no_normalize_columns:
            data[f'{col}_norm'] = data[col]
            continue
        if col in special_transforms:
            data[col] = special_transforms[col](data[col])

        data[f'{col}_norm'] = zscore_to_01(data[col])

    data['source_num'] = data['source'].map({'clip_bound': 1, 'non_determined':0.5, 'clip_unbound': 0})
    data_sorted = data.sort_values(['posterior_r', 'source_num'])

    fig, ax = plt.subplots(figsize=(7.5, 3.5))
    desired_order = [
    'TSS_proximity_norm',
    'PhastCons100_percent_norm',
    'score_phastcons100_norm', 
    'score_phylop100_norm',
    'inner_mean_logPWM_norm',
    'outer_mean_logPWM_norm',
    'AS_native_mean_inner_norm',
    'AS_fixed_mean_inner_norm',
    'Znseq_logsignal_norm',
    'prior_p_norm',
    'posterior_r_norm',
    'source_num'
    ]
    # Reorder norm_cols to match desired order
    norm_cols = [f'{col}_norm' for col in columns[:-1]] + ['source_num']
    norm_cols = [col for col in desired_order if col in norm_cols]
    matrix = data_sorted[norm_cols].values.T

    cmap = plt.matplotlib.colors.LinearSegmentedColormap.from_list('wb', 
        [(0,'white'), (0.3,'white'), (0.7,'lightblue'), (1,'blue')])
    im = ax.imshow(matrix, cmap=cmap, aspect='auto', interpolation='none')

    ax.set_xlabel("Sample")
    ax.set_title(f"{file_title}", pad=20)
    ax.set_yticks(range(len(norm_cols)))
    
    # Apply professional labels using the mapping
    professional_labels = []
    for col in norm_cols:
        original_col = col.replace('_norm','').replace('_num','')
        professional_labels.append(label_mapping.get(original_col, original_col))
    
    ax.set_yticklabels(professional_labels)
    plt.colorbar(im, ax=ax, shrink=0.6)

    if os.path.exists(motif_logo_i):
        try:
            logo_img = mpimg.imread(motif_logo_i)
            imagebox = OffsetImage(logo_img, zoom=0.2)  # Adjust zoom as needed
            ab = AnnotationBbox(imagebox, (0.85, 1.15), xycoords='axes fraction', 
                            frameon=False, box_alignment=(0.5, 0.5))
            ax.add_artist(ab)
        except:
            ax.text(0.85, 1.1, f"Motif: {rbp_df['Motif_ID'][sample_index]}", 
                transform=ax.transAxes, ha='center', fontsize=10)
    plt.tight_layout()
    plt.subplots_adjust(top=0.85)  # Make room for logo
    plt.savefig(save_heatmap_path, dpi=300, bbox_inches='tight')
    plt.show()

def heatmap_plot(data: pl.DataFrame, 
                 columns: list[str], 
                 no_normalize_columns: list[str], 
                 save_heatmap_path: str, file_title: str, motif_logo_i: str, rbp_df: pl.DataFrame, sample_index):

    # Add professional label mapping
    label_mapping = {
        'TSS_proximity': 'TSS Proximity',
        'score_phastcons100': 'PhastCons100way score',
        'score_phylop100': 'PhyloP100way score', 
        'PhastCons100_percent': 'PhastCons100way region score ',
        'inner_mean_logPWM': 'PWM score: inner region',
        'outer_mean_logPWM': 'PWM score: outer region',
        'GC_inner_pct': "GC content: inner region",
        'GC_outer_pct': 'GC content: outer region',
        'AS_native_mean_inner': 'Native AS score',
        'AS_fixed_mean_inner': 'Fixed AS score',
        's_l': 'RAD-seq signal',
        'prior_p': 'Prior',
        'posterior_r': 'Posterior',
        'source': 'CLIP label'
    }

    special_transforms = {'s_l': lambda x: np.log(x + 1)}

    data = data.select(columns).to_pandas()
    zscore_to_01 = lambda x: (stats.zscore(x) - stats.zscore(x).min()) / (stats.zscore(x).max() - stats.zscore(x).min())

    for col in columns[:-1]:
        if col in no_normalize_columns:
            data[f'{col}_norm'] = data[col]
            continue
        if col in special_transforms:
            data[col] = special_transforms[col](data[col])

        data[f'{col}_norm'] = zscore_to_01(data[col])

    data['source_num'] = data['source'].map({'clip_bound': 1, 'non_determined':0.5, 'clip_unbound': 0})
    data_sorted = data.sort_values(['posterior_r', 'source_num'])

    fig, ax = plt.subplots(figsize=(7.5, 3.5))
    desired_order = [
    'TSS_proximity_norm',
    'PhastCons100_percent_norm',
    'score_phastcons100_norm', 
    'score_phylop100_norm',
    'inner_mean_logPWM_norm',
    'outer_mean_logPWM_norm',
    'AS_native_mean_inner_norm',
    'AS_fixed_mean_inner_norm',
    's_l_norm',
    'prior_p_norm',
    'posterior_r_norm',
    'source_num'
    ]
    # Reorder norm_cols to match desired order
    norm_cols = [f'{col}_norm' for col in columns[:-1]] + ['source_num']
    norm_cols = [col for col in desired_order if col in norm_cols]
    matrix = data_sorted[norm_cols].values.T

    cmap = plt.matplotlib.colors.LinearSegmentedColormap.from_list('wb', 
        [(0,'white'), (0.3,'white'), (0.7,'lightblue'), (1,'blue')])
    im = ax.imshow(matrix, cmap=cmap, aspect='auto', interpolation='none', rasterized = True)

    ax.set_xlabel("Sample")
    ax.set_title(f"{file_title}", pad=20)
    ax.set_yticks(range(len(norm_cols)))
    
    # Apply professional labels using the mapping
    professional_labels = []
    for col in norm_cols:
        original_col = col.replace('_norm','').replace('_num','')
        professional_labels.append(label_mapping.get(original_col, original_col))
    
    ax.set_yticklabels(professional_labels)
    plt.colorbar(im, ax=ax, shrink=0.6)

    if os.path.exists(motif_logo_i):
        try:
            logo_img = mpimg.imread(motif_logo_i)
            imagebox = OffsetImage(logo_img, zoom=0.2)  # Adjust zoom as needed
            ab = AnnotationBbox(imagebox, (0.85, 1.15), xycoords='axes fraction', 
                            frameon=False, box_alignment=(0.5, 0.5))
            ax.add_artist(ab)
        except:
            ax.text(0.85, 1.1, f"Motif: {rbp_df['Motif_ID'][sample_index]}", 
                transform=ax.transAxes, ha='center', fontsize=10)
    plt.tight_layout()
    plt.subplots_adjust(top=0.85)  # Make room for logo
    plt.savefig(save_heatmap_path, dpi=300, bbox_inches='tight')
    # plt.savefig(save_heatmap_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_beta_coefficients(model, feature_columns, file_title, save_path=None):
    """Plot EM model beta coefficients"""
    import seaborn as sns
    
    beta_values = model['beta']
    
    plt.figure(figsize=(5, 3))
    sns.barplot(x=range(len(beta_values)), y=np.log(np.abs(beta_values) + 1))
    plt.xticks(range(len(beta_values)), ['intercept'] + feature_columns, rotation=35, ha='right')
    plt.title('EM Model Beta Coefficients')
    plt.ylabel('log(|beta| + 1)')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_gamma_distribution_fit(out_df, model, file_title, save_path=None):
    """Plot log(s_l) distribution with fitted gamma PDFs"""
    
    # Extract data and model parameters
    s_vals = out_df['s_l'].to_numpy()
    log_s_vals = np.log(s_vals + 1)  # Use +1 for consistency with EM input
    k0, t0 = model['gamma_params']['k0'], model['gamma_params']['theta0']
    k1, t1 = model['gamma_params']['k1'], model['gamma_params']['theta1']

    # Create histogram by source
    clip_s = np.log(out_df.filter(pl.col('source') == 'clip_bound')['s_l'].to_numpy() + 1)
    non_s = np.log(out_df.filter(pl.col('source') != 'clip_bound')['s_l'].to_numpy() + 1)

    plt.figure(figsize=(5, 3))
    plt.hist([clip_s, non_s], bins=50, alpha=0.7, 
            label=['CLIP-bound', 'not-determined'], 
            color=['red', 'blue'], density=True)

    # Plot gamma PDFs directly on log scale since EM was fitted on log(s+1)
    log_x_range = np.linspace(log_s_vals.min(), log_s_vals.max(), 1000)
    gamma_pdf_0 = stats.gamma.pdf(log_x_range, a=k0, scale=t0)
    gamma_pdf_1 = stats.gamma.pdf(log_x_range, a=k1, scale=t1)
    plt.plot(log_x_range, gamma_pdf_0, 'b--', linewidth=2, label=f'Gamma 0 (k={k0:.2f})')
    plt.plot(log_x_range, gamma_pdf_1, 'r--', linewidth=2, label=f'Gamma 1 (k={k1:.2f})')

    # Create detailed caption with statistics
    caption = f"""log(s_l + 1) Distribution with Gamma Mixture PDFs\nNon-determined mean: {np.mean(non_s):.3f} | CLIP-bound mean: {np.mean(clip_s):.3f}\nGamma 0 mean: {k0*t0:.3f} | Gamma 1 mean: {k1*t1:.3f}"""

    plt.title(caption, fontsize=10)
    plt.xlabel('log(s_l + 1)')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_gaussian_distribution_fit(out_df, model, file_title, save_path=None):
    """Plot log(s_l) distribution with fitted Gaussian PDFs"""
    
    # Extract data and model parameters
    s_vals = out_df['s_l'].to_numpy()
    log_s_vals = np.log(s_vals + 1)  # Use +1 for consistency with EM input
    mu0, sigma0 = model['gaussian_params']['mu0'], model['gaussian_params']['sigma0']
    mu1, sigma1 = model['gaussian_params']['mu1'], model['gaussian_params']['sigma1']

    # Create histogram by source
    clip_s = np.log(out_df.filter(pl.col('source') == 'clip_bound')['s_l'].to_numpy() + 1)
    non_s = np.log(out_df.filter(pl.col('source') != 'clip_bound')['s_l'].to_numpy() + 1)

    plt.figure(figsize=(7, 3))
    plt.hist([clip_s, non_s], bins=50, alpha=0.7, 
            label=['CLIP-bound', 'not-determined'], 
            color=['red', 'blue'], density=True)

    # Plot Gaussian PDFs directly on log scale since EM was fitted on log(s+1)
    log_x_range = np.linspace(log_s_vals.min(), log_s_vals.max(), 1000)
    gaussian_pdf_0 = stats.norm.pdf(log_x_range, loc=mu0, scale=sigma0)
    gaussian_pdf_1 = stats.norm.pdf(log_x_range, loc=mu1, scale=sigma1)
    plt.plot(log_x_range, gaussian_pdf_0, 'b--', linewidth=2, label=f'Gaussian 0 (μ={mu0:.2f})')
    plt.plot(log_x_range, gaussian_pdf_1, 'r--', linewidth=2, label=f'Gaussian 1 (μ={mu1:.2f})')

    # Create detailed caption with statistics
    # caption = f"""log(s_l + 1) Distribution with Gaussian Mixture PDFs\nNon-determined mean: {np.mean(non_s):.3f} | CLIP-bound mean: {np.mean(clip_s):.3f}\nGaussian 0 mean: {mu0:.3f} | Gaussian 1 mean: {mu1:.3f}"""
    

    # plt.title(caption, fontsize=10)
    import seaborn as sns
    sns.despine()
    plt.xlabel('RAD-seq score')
    plt.ylabel('Density')
    plt.legend()
    # plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_auc_em_control(out_df, file_title, save_path = None):
    from sklearn.metrics import roc_curve, auc
    eval_data = out_df.filter(pl.col('source') != "non_determined").select(['source', 'inner_mean_logPWM']).to_pandas()
    y_true = (eval_data['source'] == 'clip_bound').astype(int)
    y_scores = eval_data['inner_mean_logPWM']  # Use posterior probabilities as scores
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    return(roc_auc)

def plot_auc_em(out_df, file_title, save_path = None):
    from sklearn.metrics import roc_curve, auc
    eval_data = out_df.filter(pl.col('source') != "non_determined").select(['source', 'posterior_r']).to_pandas()

    y_true = (eval_data['source'] == 'clip_bound').astype(int)
    y_scores = eval_data['posterior_r']  # Use posterior probabilities as scores

    # Calculate ROC curve and AUC
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)

    caption = f"""Sample sizes - CLIP-bound: {sum(y_true)}, CLIP-unbound: {len(y_true) - sum(y_true)}"""

    # Create the plot
    plt.figure(figsize=(3, 3))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'{file_title}\n Sample sizes - CLIP-bound: {sum(y_true)}, CLIP-unbound: {len(y_true) - sum(y_true)}', fontsize = 9)
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    return(roc_auc)


def plot_logistic_regression_coefficients(lr_coefs, lr_intercept, feature_columns, file_title, save_path=None):
    """Plot logistic regression beta coefficients"""
    import seaborn as sns
    
    all_coefs = np.concatenate([[lr_intercept], lr_coefs])
    
    plt.figure(figsize=(5, 3))
    sns.barplot(x=range(len(all_coefs)), y=all_coefs)
    plt.xticks(range(len(all_coefs)), ['intercept', 'log(s_l + 1)'] + feature_columns, rotation=45, ha='right')
    plt.title('Logistic Regression Beta Coefficients')
    plt.ylabel('Beta Coefficient')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

# def plot_logistic_regression_coefficients(lr_coefs, lr_intercept, feature_columns, file_title, save_path=None):

def plot_logistic_regression_roc(fpr, tpr, auc_score, n_samples, file_title, save_path=None):
    """Plot logistic regression ROC curve"""
    
    plt.figure(figsize=(3, 3))
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC curve (AUC = {auc_score:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Logistic Regression ROC Curve\n{file_title} (n={n_samples})')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def save_analysis_results_easy(model, source_coefs, source_coefs_s_l, source_intercept, source_intercept_s_l, feature_columns, roc_auc, roc_auc_control, auc_source, auc_source_s_l,
                         file_title, out_df, motif_df=None, output_dir='../logs'):
    """
    Save beta coefficients and metadata from EM and logistic regression analysis
    
    Parameters:
    - model: EM model results dictionary
    - source_coefs: Logistic regression coefficients
    - source_intercept: Logistic regression intercept
    - source_coefs_s_l: Logistic regression coefficients with s_l in training data
    - source_intercept_s_l: Logistic regression intercept with s_l in training data
    - feature_columns: List of feature names
    - roc_auc: AUC score from EM model
    - roc_auc_control: AUC score from using inner_log from using inner_logPWM as control.
    - auc_source: AUC score from logistic regression
    - auc_source_s_l: AUC score from logistic regression with s_l in training data
    - file_title: Title for output files
    - out_df: Output dataframe with source labels (must contain 'posterior_r' and 'source')
    - motif_df: Original motif dataframe with 'source' to count each category (optional)
    - output_dir: Directory to save files (default: '../logs')
    
    Returns:
    - beta_results_df_s_l, beta_results_df, metadata_df
    """
    import os
    import polars as pl
    import numpy as np
    from sklearn.metrics import (
        average_precision_score, precision_recall_curve,
        roc_curve, f1_score
    )

    def safe_counts(df_pl):
        if df_pl is None:
            return 0, 0, 0
        vc = df_pl['source'].value_counts().to_pandas().set_index('source')
        get = lambda k: (vc.loc[k, 'count'] if k in vc.index else 0)
        return get('clip_bound'), get('clip_unbound'), get('non_determined')

    def best_f1(y_true, y_score):
        # Compute best F1 and corresponding threshold
        prec, rec, thr = precision_recall_curve(y_true, y_score)
        # precision_recall_curve returns len(thr) = len(prec) - 1
        f1_all = 2 * prec[:-1] * rec[:-1] / (prec[:-1] + rec[:-1] + 1e-12)
        if len(f1_all) == 0 or np.all(np.isnan(f1_all)):
            return 0.0, 0.5
        idx = int(np.nanargmax(f1_all))
        return float(f1_all[idx]), float(thr[idx])

    def precision_at_fpr(y_true, y_score, target_fpr):
        # Pick the operating point with max TPR under FPR <= target_fpr
        fpr, tpr, thr = roc_curve(y_true, y_score)
        if len(fpr) == 0:
            return 0.0, 0.0, 0.0, 0.5
        valid = fpr <= target_fpr
        if valid.any():
            sub_idx = np.argmax(tpr[valid])
            idx = np.flatnonzero(valid)[sub_idx]
        else:
            # If no threshold meets target, choose closest FPR
            idx = int(np.argmin(np.abs(fpr - target_fpr)))
        thr_sel = float(thr[idx])
        y_pred = (y_score >= thr_sel).astype(int)
        tp = int(((y_pred == 1) & (y_true == 1)).sum())
        fp = int(((y_pred == 1) & (y_true == 0)).sum())
        prec = float(tp / (tp + fp + 1e-12))
        return prec, float(fpr[idx]), float(tpr[idx]), thr_sel

    # Extract beta coefficients
    beta_em = model['beta']
    beta_lr = np.concatenate([[source_intercept], source_coefs])
    beta_lr_s_l = np.concatenate([[source_intercept_s_l], source_coefs_s_l])
    
    # Calculate site counts for each category in out_df (used previously)
    site_counts = out_df['source'].value_counts().to_pandas().set_index('source')
    n_clip_bound = site_counts.loc['clip_bound', 'count'] if 'clip_bound' in site_counts.index else 0
    n_clip_unbound = site_counts.loc['clip_unbound', 'count'] if 'clip_unbound' in site_counts.index else 0
    n_non_determined = site_counts.loc['non_determined', 'count'] if 'non_determined' in site_counts.index else 0

    # Also count categories in motif_df if provided
    m_clip_bound, m_clip_unbound, m_non_determined = safe_counts(motif_df)

    # Compute EM metrics on labeled examples only
    filt = out_df.filter(pl.col('source') != 'non_determined')
    if filt.height > 0:
        y_true = (filt['source'] == 'clip_bound').to_numpy().astype(int)
        y_score = filt['posterior_r'].to_numpy()
        try:
            ap_em = float(average_precision_score(y_true, y_score))
        except Exception:
            ap_em = float('nan')
        try:
            f1_best, f1_thr = best_f1(y_true, y_score)
        except Exception:
            f1_best, f1_thr = float('nan'), float('nan')
        # Precision at FPR targets
        try:
            prec_10_fpr, act_fpr_10, act_tpr_10, thr_10 = precision_at_fpr(y_true, y_score, 0.10)
        except Exception:
            prec_10_fpr, act_fpr_10, act_tpr_10, thr_10 = float('nan'), float('nan'), float('nan'), float('nan')
        try:
            prec_1_fpr, act_fpr_1, act_tpr_1, thr_1 = precision_at_fpr(y_true, y_score, 0.01)
        except Exception:
            prec_1_fpr, act_fpr_1, act_tpr_1, thr_1 = float('nan'), float('nan'), float('nan'), float('nan')
    else:
        ap_em = f1_best = f1_thr = float('nan')
        prec_10_fpr = act_fpr_10 = act_tpr_10 = thr_10 = float('nan')
        prec_1_fpr = act_fpr_1 = act_tpr_1 = thr_1 = float('nan')
    
    
    # Create dataframe with beta coefficients
    beta_results_data = {
        'feature': ['intercept'] + feature_columns,
        'beta_em': beta_em,
        'beta_lr': beta_lr,
    }
    beta_results_data_s_l = {
        'feature': ['intercept', "s_l"] + feature_columns,
        'beta_lr_s_l': beta_lr_s_l
    }
    beta_results_df = pl.DataFrame(beta_results_data)
    beta_results_df_s_l = pl.DataFrame(beta_results_data_s_l)
    
    # Create metadata dataframe with requested metrics and counts
    metadata_df = pl.DataFrame({
        'file_title': [file_title],
        # EM ROC/PR metrics
        'auc_control': [roc_auc_control],
        'auc_em': [roc_auc],
        'ap_em': [ap_em],
        'f1_best_em': [f1_best],
        'f1_best_threshold_em': [f1_thr],
        # Precision at FPR operating points
        'precision_at_10pct_fpr_em': [prec_10_fpr],
        'actual_fpr_at_10pct_target': [act_fpr_10],
        'tpr_at_10pct_fpr_em': [act_tpr_10],
        'threshold_at_10pct_fpr_em': [thr_10],
        'precision_at_1pct_fpr_em': [prec_1_fpr],
        'actual_fpr_at_1pct_target': [act_fpr_1],
        'tpr_at_1pct_fpr_em': [act_tpr_1],
        'threshold_at_1pct_fpr_em': [thr_1],
        # Logistic regression AUCs
        'auc_lr': [auc_source],
        'auc_lr_sl': [auc_source_s_l],
        # Counts from out_df (post-EM)
        'n_clip_bound_out_df': [n_clip_bound],
        'n_clip_unbound_out_df': [n_clip_unbound],
        'n_non_determined_out_df': [n_non_determined],
        # Counts from motif_df (requested)
        'n_clip_bound_motif_df': [m_clip_bound],
        'n_clip_unbound_motif_df': [m_clip_unbound],
        'n_non_determined_motif_df': [m_non_determined],
    })
    
    # Save
    os.makedirs(output_dir, exist_ok=True)
    beta_results_df_s_l.write_csv(f'{output_dir}/{file_title}_beta_coefficients_s_l.tsv', separator='\t')
    beta_results_df.write_csv(f'{output_dir}/{file_title}_beta_coefficients.tsv', separator='\t')
    metadata_df.write_csv(f'{output_dir}/{file_title}_metadata.tsv', separator='\t')
    
    return beta_results_df_s_l, beta_results_df, metadata_df

def save_analysis_results(model, source_coefs, source_coefs_s_l, source_intercept, source_intercept_s_l, feature_columns, roc_auc, roc_auc_control, auc_source, auc_source_s_l,
                         file_title, out_df, test_dict, motif_df=None, output_dir='../logs'):
    """
    Save beta coefficients and metadata from EM and logistic regression analysis
    
    Parameters:
    - model: EM model results dictionary
    - source_coefs: Logistic regression coefficients
    - source_intercept: Logistic regression intercept
    - source_coefs_s_l: Logistic regression coefficients with s_l in training data
    - source_intercept_s_l: Logistic regression intercept with s_l in training data
    - feature_columns: List of feature names
    - roc_auc: AUC score from EM model
    - roc_auc_control: AUC score from using inner_log from using inner_logPWM as control.
    - auc_source: AUC score from logistic regression
    - auc_source_s_l: AUC score from logistic regression with s_l in training data
    - file_title: Title for output files
    - out_df: Output dataframe with source labels (must contain 'posterior_r' and 'source')
    - test_dict: Q-Q validation test results
    - motif_df: Original motif dataframe with 'source' to count each category (optional)
    - output_dir: Directory to save files (default: '../logs')
    
    Returns:
    - beta_results_df_s_l, beta_results_df, metadata_df
    """
    import os
    import polars as pl
    import numpy as np
    from sklearn.metrics import (
        average_precision_score, precision_recall_curve,
        roc_curve, f1_score
    )

    def safe_counts(df_pl):
        if df_pl is None:
            return 0, 0, 0
        vc = df_pl['source'].value_counts().to_pandas().set_index('source')
        get = lambda k: (vc.loc[k, 'count'] if k in vc.index else 0)
        return get('clip_bound'), get('clip_unbound'), get('non_determined')

    def best_f1(y_true, y_score):
        # Compute best F1 and corresponding threshold
        prec, rec, thr = precision_recall_curve(y_true, y_score)
        # precision_recall_curve returns len(thr) = len(prec) - 1
        f1_all = 2 * prec[:-1] * rec[:-1] / (prec[:-1] + rec[:-1] + 1e-12)
        if len(f1_all) == 0 or np.all(np.isnan(f1_all)):
            return 0.0, 0.5
        idx = int(np.nanargmax(f1_all))
        return float(f1_all[idx]), float(thr[idx])

    def precision_at_fpr(y_true, y_score, target_fpr):
        # Pick the operating point with max TPR under FPR <= target_fpr
        fpr, tpr, thr = roc_curve(y_true, y_score)
        if len(fpr) == 0:
            return 0.0, 0.0, 0.0, 0.5
        valid = fpr <= target_fpr
        if valid.any():
            sub_idx = np.argmax(tpr[valid])
            idx = np.flatnonzero(valid)[sub_idx]
        else:
            # If no threshold meets target, choose closest FPR
            idx = int(np.argmin(np.abs(fpr - target_fpr)))
        thr_sel = float(thr[idx])
        y_pred = (y_score >= thr_sel).astype(int)
        tp = int(((y_pred == 1) & (y_true == 1)).sum())
        fp = int(((y_pred == 1) & (y_true == 0)).sum())
        prec = float(tp / (tp + fp + 1e-12))
        return prec, float(fpr[idx]), float(tpr[idx]), thr_sel

    # Extract beta coefficients
    beta_em = model['beta']
    beta_lr = np.concatenate([[source_intercept], source_coefs])
    beta_lr_s_l = np.concatenate([[source_intercept_s_l], source_coefs_s_l])
    
    # Calculate site counts for each category in out_df (used previously)
    site_counts = out_df['source'].value_counts().to_pandas().set_index('source')
    n_clip_bound = site_counts.loc['clip_bound', 'count'] if 'clip_bound' in site_counts.index else 0
    n_clip_unbound = site_counts.loc['clip_unbound', 'count'] if 'clip_unbound' in site_counts.index else 0
    n_non_determined = site_counts.loc['non_determined', 'count'] if 'non_determined' in site_counts.index else 0

    # Also count categories in motif_df if provided
    m_clip_bound, m_clip_unbound, m_non_determined = safe_counts(motif_df)

    # Compute EM metrics on labeled examples only
    filt = out_df.filter(pl.col('source') != 'non_determined')
    if filt.height > 0:
        y_true = (filt['source'] == 'clip_bound').to_numpy().astype(int)
        y_score = filt['posterior_r'].to_numpy()
        try:
            ap_em = float(average_precision_score(y_true, y_score))
        except Exception:
            ap_em = float('nan')
        try:
            f1_best, f1_thr = best_f1(y_true, y_score)
        except Exception:
            f1_best, f1_thr = float('nan'), float('nan')
        # Precision at FPR targets
        try:
            prec_10_fpr, act_fpr_10, act_tpr_10, thr_10 = precision_at_fpr(y_true, y_score, 0.10)
        except Exception:
            prec_10_fpr, act_fpr_10, act_tpr_10, thr_10 = float('nan'), float('nan'), float('nan'), float('nan')
        try:
            prec_1_fpr, act_fpr_1, act_tpr_1, thr_1 = precision_at_fpr(y_true, y_score, 0.01)
        except Exception:
            prec_1_fpr, act_fpr_1, act_tpr_1, thr_1 = float('nan'), float('nan'), float('nan'), float('nan')
    else:
        ap_em = f1_best = f1_thr = float('nan')
        prec_10_fpr = act_fpr_10 = act_tpr_10 = thr_10 = float('nan')
        prec_1_fpr = act_fpr_1 = act_tpr_1 = thr_1 = float('nan')
    
    # Extract Q-Q deviation statistics from test_dict
    qq_clip_bound_gamma = test_dict.get('clip_bound', {}).get('Gamma', {}).get('qq_deviation', None)
    qq_clip_bound_normal = test_dict.get('clip_bound', {}).get('Normal', {}).get('qq_deviation', None)
    qq_clip_bound_student = test_dict.get('clip_bound', {}).get('Student-t', {}).get('qq_deviation', None)
    qq_clip_unbound_gamma = test_dict.get('clip_unbound', {}).get('Gamma', {}).get('qq_deviation', None)
    qq_clip_unbound_normal = test_dict.get('clip_unbound', {}).get('Normal', {}).get('qq_deviation', None)
    qq_clip_unbound_student = test_dict.get('clip_unbound', {}).get('Student-t', {}).get('qq_deviation', None)
    
    # Create dataframe with beta coefficients
    beta_results_data = {
        'feature': ['intercept'] + feature_columns,
        'beta_em': beta_em,
        'beta_lr': beta_lr,
    }
    beta_results_data_s_l = {
        'feature': ['intercept', "s_l"] + feature_columns,
        'beta_lr_s_l': beta_lr_s_l
    }
    beta_results_df = pl.DataFrame(beta_results_data)
    beta_results_df_s_l = pl.DataFrame(beta_results_data_s_l)
    
    # Create metadata dataframe with requested metrics and counts
    metadata_df = pl.DataFrame({
        'file_title': [file_title],
        # EM ROC/PR metrics
        'auc_control': [roc_auc_control],
        'auc_em': [roc_auc],
        'ap_em': [ap_em],
        'f1_best_em': [f1_best],
        'f1_best_threshold_em': [f1_thr],
        # Precision at FPR operating points
        'precision_at_10pct_fpr_em': [prec_10_fpr],
        'actual_fpr_at_10pct_target': [act_fpr_10],
        'tpr_at_10pct_fpr_em': [act_tpr_10],
        'threshold_at_10pct_fpr_em': [thr_10],
        'precision_at_1pct_fpr_em': [prec_1_fpr],
        'actual_fpr_at_1pct_target': [act_fpr_1],
        'tpr_at_1pct_fpr_em': [act_tpr_1],
        'threshold_at_1pct_fpr_em': [thr_1],
        # Logistic regression AUCs
        'auc_lr': [auc_source],
        'auc_lr_sl': [auc_source_s_l],
        # Counts from out_df (post-EM)
        'n_clip_bound_out_df': [n_clip_bound],
        'n_clip_unbound_out_df': [n_clip_unbound],
        'n_non_determined_out_df': [n_non_determined],
        # Counts from motif_df (requested)
        'n_clip_bound_motif_df': [m_clip_bound],
        'n_clip_unbound_motif_df': [m_clip_unbound],
        'n_non_determined_motif_df': [m_non_determined],
        # QQ stats
        'qq_clip_bound_gamma': [qq_clip_bound_gamma],
        'qq_clip_bound_normal': [qq_clip_bound_normal],
        'qq_clip_bound_student': [qq_clip_bound_student],
        'qq_clip_unbound_gamma': [qq_clip_unbound_gamma],
        'qq_clip_unbound_normal': [qq_clip_unbound_normal],
        'qq_clip_unbound_student': [qq_clip_unbound_student]
    })
    
    # Save
    os.makedirs(output_dir, exist_ok=True)
    beta_results_df_s_l.write_csv(f'{output_dir}/{file_title}_beta_coefficients_s_l.tsv', separator='\t')
    beta_results_df.write_csv(f'{output_dir}/{file_title}_beta_coefficients.tsv', separator='\t')
    metadata_df.write_csv(f'{output_dir}/{file_title}_metadata.tsv', separator='\t')
    
    return beta_results_df_s_l, beta_results_df, metadata_df

# ...existing code...

def change_label(out_df):
    diff_phylop = out_df.select( (
            pl.when(pl.col("posterior_r") > 0.9)
            .then(pl.col("score_phylop100"))
            .mean()
        - pl.when(pl.col("posterior_r") < 0.1)
            .then(pl.col("score_phylop100"))
            .mean()
        ).alias("mean_diff")
    ).item()
    diff_phastcons = out_df.select( (
            pl.when(pl.col("posterior_r") > 0.9)
            .then(pl.col("score_phastcons100"))
            .mean()
        - pl.when(pl.col("posterior_r") < 0.1)
            .then(pl.col("score_phastcons100"))
            .mean()
        ).alias("mean_diff")
    ).item()
    if (diff_phylop > 0) and (diff_phastcons > 0):
        print(f"Both PhyloP and PhastCons show positive selection: {diff_phylop}, {diff_phastcons}")
        return out_df
    if diff_phastcons < 0:
        print(f"PhastCons shows negative selection: {diff_phastcons}, diff_phylop is {diff_phylop}, change the label")
        out_df = out_df.with_columns(
            (1 - pl.col("posterior_r")).alias("posterior_r"), 
            (1 - pl.col("prior_p")).alias("prior_p")
        )
        return out_df



# Add this function to em_utils.py
# Update the existing plot_precision_recall_em function with AP calculation

def plot_precision_recall_em(out_df, file_title, save_path=None):
    """Plot precision-recall curve with AP in legend."""
    from sklearn.metrics import precision_recall_curve, average_precision_score
    import matplotlib.pyplot as plt
    
    # Filter out non-determined sites
    filtered_df = out_df.filter(out_df['source'] != 'non_determined')
    
    # Get true labels and predicted probabilities
    y_true = (filtered_df['source'] == 'clip_bound').to_numpy().astype(int)
    y_scores = filtered_df['posterior_r'].to_numpy()
    
    # Calculate precision-recall curve and AP
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    avg_precision = average_precision_score(y_true, y_scores)
    baseline = np.sum(y_true) / len(y_true)  # Random classifier AP
    
    # Create the plot
    plt.figure(figsize=(3, 3))
    plt.plot(recall, precision, 'b-', linewidth=2, 
             label=f'Model (AP = {avg_precision:.3f})')
    plt.axhline(y=baseline, color='r', linestyle='--', 
                label=f'Random (AP = {baseline:.3f})')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve\n{file_title}')
    plt.legend(loc="lower left")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
    
    # Return AP score
    return avg_precision
