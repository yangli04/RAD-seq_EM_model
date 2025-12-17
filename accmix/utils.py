from __future__ import annotations

import os
from typing import Dict, Any, List, Tuple

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import numpy as np
import polars as pl
from scipy import stats
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score


def compute_qq_validate_distribution(
    out_df: pl.DataFrame,
    variable: str = "s_l",
    addition: float = 1.0,
    save_figure_path: str = "plots/distribution.png",
) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """Distribution diagnostics for log-transformed variable by source label.

    Returns a nested dict of summary statistics per source and distribution
    family (Gamma, Normal, Student-t).
    """

    sources = [s for s in out_df["source"].unique().to_list() if s != "non_determined"]
    distributions = [("Gamma", stats.gamma), ("Normal", stats.norm), ("Student-t", stats.t)]

    fig, axes = plt.subplots(len(sources), 3, figsize=(8, 2 * len(sources)))
    if len(sources) == 1:
        axes = axes.reshape(1, -1)

    results: Dict[str, Dict[str, Dict[str, Any]]] = {}

    for i, source in enumerate(sources):
        log_data = np.log(out_df.filter(pl.col("source") == source)[variable].to_numpy() + addition)
        results[source] = {}

        for j, (name, dist) in enumerate(distributions):
            if name == "Gamma":
                params = dist.fit(log_data, floc=0)
            else:
                params = dist.fit(log_data)

            stats.probplot(log_data, dist=dist, sparams=params, plot=axes[i, j])

            theoretical = axes[i, j].get_lines()[1].get_xdata()
            empirical = axes[i, j].get_lines()[1].get_ydata()
            r2 = float(np.corrcoef(theoretical, empirical)[0, 1] ** 2)
            ks_stat, ks_p = stats.kstest(log_data, lambda x: dist.cdf(x, *params))
            qq_deviation = float(np.mean(np.abs(empirical - theoretical)))

            axes[i, j].set_title(f"{source} - {name}\nR²={r2:.3f}, QQ-dev={qq_deviation:.3f}\nKS p={ks_p:.3f}")
            axes[i, j].grid(True, alpha=0.3)

            results[source][name] = {
                "r2": r2,
                "ks_p": float(ks_p),
                "qq_deviation": qq_deviation,
                "params": params,
                "n_samples": len(log_data),
            }

    os.makedirs(os.path.dirname(save_figure_path) or ".", exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_figure_path)
    plt.close(fig)
    return results


def heatmap_plot(
    data: pl.DataFrame,
    columns: List[str],
    no_normalize_columns: List[str],
    save_heatmap_path: str,
    file_title: str,
    motif_logo: str | None = None,
) -> None:
    """Plot feature heatmap ordered by posterior_r and source label."""

    label_mapping = {
        "TSS_proximity": "TSS Proximity",
        "score_phastcons100": "PhastCons100way score",
        "score_phylop100": "PhyloP100way score",
        "PhastCons100_percent": "PhastCons100way region score",
        "inner_mean_logPWM": "PWM score: inner region",
        "outer_mean_logPWM": "PWM score: outer region",
        "GC_inner_pct": "GC content: inner region",
        "GC_outer_pct": "GC content: outer region",
        "s_l": "RAD-seq signal",
        "prior_p": "Prior",
        "posterior_r": "Posterior",
        "source": "CLIP label",
    }

    special_transforms = {"s_l": lambda x: np.log(x + 1)}

    pdf = data.select(columns).to_pandas()

    def zscore_to_01(x: np.ndarray) -> np.ndarray:
        z = stats.zscore(x)
        return (z - np.nanmin(z)) / (np.nanmax(z) - np.nanmin(z) + 1e-12)

    for col in columns[:-1]:
        if col in no_normalize_columns:
            pdf[f"{col}_norm"] = pdf[col]
            continue
        if col in special_transforms:
            pdf[col] = special_transforms[col](pdf[col])
        pdf[f"{col}_norm"] = zscore_to_01(pdf[col].to_numpy())

    pdf["source_num"] = pdf["source"].map({"clip_bound": 1.0, "non_determined": 0.5, "clip_unbound": 0.0})
    pdf_sorted = pdf.sort_values(["posterior_r", "source_num"])

    desired_order = [
        "TSS_proximity_norm",
        "PhastCons100_percent_norm",
        "score_phastcons100_norm",
        "score_phylop100_norm",
        "inner_mean_logPWM_norm",
        "outer_mean_logPWM_norm",
        "s_l_norm",
        "prior_p_norm",
        "posterior_r_norm",
        "source_num",
    ]

    norm_cols = [f"{col}_norm" for col in columns[:-1]] + ["source_num"]
    norm_cols = [c for c in desired_order if c in norm_cols]
    matrix = pdf_sorted[norm_cols].values.T

    cmap = plt.matplotlib.colors.LinearSegmentedColormap.from_list(
        "wb",
        [(0, "white"), (0.3, "white"), (0.7, "lightblue"), (1, "blue")],
    )
    fig, ax = plt.subplots(figsize=(7.5, 3.5))
    im = ax.imshow(matrix, cmap=cmap, aspect="auto", interpolation="none", rasterized=True)

    ax.set_xlabel("Sample")
    ax.set_title(file_title, pad=20)
    ax.set_yticks(range(len(norm_cols)))

    professional_labels: List[str] = []
    for col in norm_cols:
        original_col = col.replace("_norm", "").replace("_num", "")
        professional_labels.append(label_mapping.get(original_col, original_col))
    ax.set_yticklabels(professional_labels)
    plt.colorbar(im, ax=ax, shrink=0.6)

    if motif_logo and os.path.exists(motif_logo):
        try:
            logo_img = mpimg.imread(motif_logo)
            imagebox = OffsetImage(logo_img, zoom=0.2)
            ab = AnnotationBbox(
                imagebox,
                (0.85, 1.15),
                xycoords="axes fraction",
                frameon=False,
                box_alignment=(0.5, 0.5),
            )
            ax.add_artist(ab)
        except Exception:
            pass

    os.makedirs(os.path.dirname(save_heatmap_path) or ".", exist_ok=True)
    plt.tight_layout()
    plt.subplots_adjust(top=0.85)
    plt.savefig(save_heatmap_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_em_auc(out_df: pl.DataFrame, file_title: str, save_path: str | None = None) -> float:
    """ROC AUC for EM posterior_r vs CLIP-bound / clip_unbound labels."""

    eval_df = out_df.filter(pl.col("source") != "non_determined").select(["source", "posterior_r"]).to_pandas()
    y_true = (eval_df["source"] == "clip_bound").astype(int).to_numpy()
    y_scores = eval_df["posterior_r"].to_numpy()

    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(3, 3))
    ax.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC (AUC = {roc_auc:.3f})")
    ax.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--", label="Random")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.05)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(file_title)
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return float(roc_auc)


def plot_em_auc_control(out_df: pl.DataFrame) -> float:
    """Baseline AUC using inner_mean_logPWM as the score."""

    eval_df = (
        out_df.filter(pl.col("source") != "non_determined")
        .select(["source", "inner_mean_logPWM"])
        .to_pandas()
    )
    y_true = (eval_df["source"] == "clip_bound").astype(int).to_numpy()
    y_scores = eval_df["inner_mean_logPWM"].to_numpy()

    fpr, tpr, _ = roc_curve(y_true, y_scores)
    return float(auc(fpr, tpr))


def change_label(out_df: pl.DataFrame) -> pl.DataFrame:
    """Flip posterior/prior labels if conservation suggests negative selection.

    If PhastCons-based difference is negative while PhyloP is not strongly
    positive, posterior_r and prior_p are inverted.
    """

    diff_phylop = (
        out_df.select(
            (
                pl.when(pl.col("posterior_r") > 0.9)
                .then(pl.col("score_phylop100"))
                .mean()
                - pl.when(pl.col("posterior_r") < 0.1)
                .then(pl.col("score_phylop100"))
                .mean()
            ).alias("mean_diff")
        ).item()
    )
    diff_phastcons = (
        out_df.select(
            (
                pl.when(pl.col("posterior_r") > 0.9)
                .then(pl.col("score_phastcons100"))
                .mean()
                - pl.when(pl.col("posterior_r") < 0.1)
                .then(pl.col("score_phastcons100"))
                .mean()
            ).alias("mean_diff")
        ).item()
    )

    if (diff_phylop > 0) and (diff_phastcons > 0):
        return out_df
    if diff_phastcons < 0:
        return out_df.with_columns(
            (1 - pl.col("posterior_r")).alias("posterior_r"),
            (1 - pl.col("prior_p")).alias("prior_p"),
        )
    return out_df


def save_analysis_results(
    model: Dict[str, Any] | None,
    source_coefs: np.ndarray,
    source_coefs_s_l: np.ndarray,
    source_intercept: float,
    source_intercept_s_l: float,
    feature_columns: List[str],
    roc_auc: float,
    roc_auc_control: float,
    auc_source: float,
    auc_source_s_l: float,
    file_title: str,
    out_df: pl.DataFrame,
    test_dict: Dict[str, Any] | None,
    motif_df: pl.DataFrame | None = None,
    output_dir: str = "logs",
) -> Tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Persist beta coefficients and key evaluation metrics to TSVs."""

    os.makedirs(output_dir, exist_ok=True)

    def safe_counts(df_pl: pl.DataFrame | None) -> Tuple[int, int, int]:
        if df_pl is None:
            return 0, 0, 0
        vc = df_pl["source"].value_counts().to_pandas().set_index("source")
        def get(k: str) -> int:
            return int(vc.loc[k, "count"]) if k in vc.index else 0
        return get("clip_bound"), get("clip_unbound"), get("non_determined")

    def best_f1(y_true: np.ndarray, y_score: np.ndarray) -> Tuple[float, float]:
        prec, rec, thr = precision_recall_curve(y_true, y_score)
        f1_all = 2 * prec[:-1] * rec[:-1] / (prec[:-1] + rec[:-1] + 1e-12)
        if len(f1_all) == 0 or np.all(np.isnan(f1_all)):
            return 0.0, 0.5
        idx = int(np.nanargmax(f1_all))
        return float(f1_all[idx]), float(thr[idx])

    def precision_at_fpr(
        y_true: np.ndarray, y_score: np.ndarray, target_fpr: float
    ) -> Tuple[float, float, float, float]:
        fpr, tpr, thr = roc_curve(y_true, y_score)
        if len(fpr) == 0:
            return 0.0, 0.0, 0.0, 0.5
        valid = fpr <= target_fpr
        if valid.any():
            sub_idx = np.argmax(tpr[valid])
            idx = np.flatnonzero(valid)[sub_idx]
        else:
            idx = int(np.argmin(np.abs(fpr - target_fpr)))
        thr_sel = float(thr[idx])
        y_pred = (y_score >= thr_sel).astype(int)
        p = int(((y_pred == 1) & (y_true == 1)).sum())
        fp = int(((y_pred == 1) & (y_true == 0)).sum())
        prec = float(p / (p + fp + 1e-12))
        return prec, float(fpr[idx]), float(tpr[idx]), thr_sel

    if model is not None and isinstance(model, dict) and "beta" in model:
        beta_em = model["beta"]
    else:
        # Fallback: fill with NaNs when model params are unavailable
        beta_em = [float("nan")] * (len(feature_columns) + 1)
    beta_lr = np.concatenate([[source_intercept], source_coefs])
    beta_lr_s_l = np.concatenate([[source_intercept_s_l], source_coefs_s_l])

    site_counts = out_df["source"].value_counts().to_pandas().set_index("source")
    n_clip_bound = int(site_counts.loc["clip_bound", "count"]) if "clip_bound" in site_counts.index else 0
    n_clip_unbound = int(site_counts.loc["clip_unbound", "count"]) if "clip_unbound" in site_counts.index else 0
    n_non_determined = int(site_counts.loc["non_determined", "count"]) if "non_determined" in site_counts.index else 0

    m_clip_bound, m_clip_unbound, m_non_determined = safe_counts(motif_df)

    filt = out_df.filter(pl.col("source") != "non_determined")
    if filt.height > 0:
        y_true = (filt["source"] == "clip_bound").to_numpy().astype(int)
        y_score = filt["posterior_r"].to_numpy()
        try:
            ap_em = float(average_precision_score(y_true, y_score))
        except Exception:
            ap_em = float("nan")
        try:
            f1_best, f1_thr = best_f1(y_true, y_score)
        except Exception:
            f1_best, f1_thr = float("nan"), float("nan")
        try:
            prec_10_fpr, act_fpr_10, act_tpr_10, thr_10 = precision_at_fpr(y_true, y_score, 0.10)
        except Exception:
            prec_10_fpr = act_fpr_10 = act_tpr_10 = thr_10 = float("nan")
        try:
            prec_1_fpr, act_fpr_1, act_tpr_1, thr_1 = precision_at_fpr(y_true, y_score, 0.01)
        except Exception:
            prec_1_fpr = act_fpr_1 = act_tpr_1 = thr_1 = float("nan")
    else:
        ap_em = f1_best = f1_thr = float("nan")
        prec_10_fpr = act_fpr_10 = act_tpr_10 = thr_10 = float("nan")
        prec_1_fpr = act_fpr_1 = act_tpr_1 = thr_1 = float("nan")

    qq_stats = {
        "clip_bound": {},
        "clip_unbound": {},
    }
    if test_dict is not None:
        for src in ["clip_bound", "clip_unbound"]:
            for dist_name in ["Gamma", "Normal", "Student-t"]:
                qq_stats[src][dist_name] = (
                    test_dict.get(src, {})
                    .get(dist_name, {})
                    .get("qq_deviation", None)
                )

    beta_results_df = pl.DataFrame(
        {
            "feature": ["intercept"] + feature_columns,
            "beta_em": beta_em,
            "beta_lr": beta_lr,
        }
    )
    beta_results_df_s_l = pl.DataFrame(
        {
            "feature": ["intercept", "s_l"] + feature_columns,
            "beta_lr_s_l": beta_lr_s_l,
        }
    )

    metadata_df = pl.DataFrame(
        {
            "file_title": [file_title],
            "auc_control": [roc_auc_control],
            "auc_em": [roc_auc],
            "ap_em": [ap_em],
            "f1_best_em": [f1_best],
            "f1_best_threshold_em": [f1_thr],
            "precision_at_10pct_fpr_em": [prec_10_fpr],
            "actual_fpr_at_10pct_target": [act_fpr_10],
            "tpr_at_10pct_fpr_em": [act_tpr_10],
            "threshold_at_10pct_fpr_em": [thr_10],
            "precision_at_1pct_fpr_em": [prec_1_fpr],
            "actual_fpr_at_1pct_target": [act_fpr_1],
            "tpr_at_1pct_fpr_em": [act_tpr_1],
            "threshold_at_1pct_fpr_em": [thr_1],
            "auc_lr": [auc_source],
            "auc_lr_sl": [auc_source_s_l],
            "n_clip_bound_out_df": [n_clip_bound],
            "n_clip_unbound_out_df": [n_clip_unbound],
            "n_non_determined_out_df": [n_non_determined],
            "n_clip_bound_motif_df": [m_clip_bound],
            "n_clip_unbound_motif_df": [m_clip_unbound],
            "n_non_determined_motif_df": [m_non_determined],
            "qq_clip_bound_gamma": [qq_stats["clip_bound"].get("Gamma")],
            "qq_clip_bound_normal": [qq_stats["clip_bound"].get("Normal")],
            "qq_clip_bound_student": [qq_stats["clip_bound"].get("Student-t")],
            "qq_clip_unbound_gamma": [qq_stats["clip_unbound"].get("Gamma")],
            "qq_clip_unbound_normal": [qq_stats["clip_unbound"].get("Normal")],
            "qq_clip_unbound_student": [qq_stats["clip_unbound"].get("Student-t")],
        }
    )

    beta_results_df_s_l.write_csv(os.path.join(output_dir, f"{file_title}_beta_coefficients_s_l.tsv"), separator="\t")
    beta_results_df.write_csv(os.path.join(output_dir, f"{file_title}_beta_coefficients.tsv"), separator="\t")
    metadata_df.write_csv(os.path.join(output_dir, f"{file_title}_metadata.tsv"), separator="\t")

    return beta_results_df_s_l, beta_results_df, metadata_df


def plot_precision_recall_em(
    out_df: pl.DataFrame, file_title: str, save_path: str | None = None
) -> float:
    """Precision–recall curve and average precision for EM posteriors."""

    filtered_df = out_df.filter(pl.col("source") != "non_determined")
    y_true = (filtered_df["source"] == "clip_bound").to_numpy().astype(int)
    y_scores = filtered_df["posterior_r"].to_numpy()

    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    avg_precision = average_precision_score(y_true, y_scores)
    baseline = float(np.sum(y_true) / len(y_true)) if len(y_true) else 0.0

    fig, ax = plt.subplots(figsize=(3, 3))
    ax.plot(recall, precision, "b-", linewidth=2, label=f"Model (AP = {avg_precision:.3f})")
    ax.axhline(y=baseline, color="r", linestyle="--", label=f"Random (AP = {baseline:.3f})")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.05)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"Precision–Recall Curve\n{file_title}")
    ax.legend(loc="lower left")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return float(avg_precision)
