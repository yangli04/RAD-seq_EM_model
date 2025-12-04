from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
import pyranges as pr
import polars as pl
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc

from accmix.model_core import read_polars_input
from accmix import utils as utils


def _load_config(path: str) -> Dict[str, Any]:
    p = Path(path)
    text = p.read_text()
    # Simple heuristic: JSON only for now (can extend to TOML/YAML later)
    return json.loads(text)


def run_evaluation(config_path: str) -> None:
    """Run evaluation pipeline based on a JSON config.

        The config schema (JSON):
        {
            "input_data_parquet": "path/to/RBP_Motif.model.parquet",  # fitted model data (motif_files from run_mixture_model.py)
            "clipseq_bed": "path/to/clipseq.bed",
            "pipseq_parquet": "path/to/PIPseq.parquet",
            "rbp_name": "RBPName",
            "motif_id": "M00001",
            "motif_logo": "path/to/logo.png",
            "output_root": "results",                                  # for plots/logs
            "model_json": "results/RBP_Motif.model.json",              # trained model parameters
            "score_phastcons100_threshold": 1.0,
            "motif_range": 50
        }
    """

    cfg = _load_config(config_path)
    # Input data produced by the model step ("motif_files" in run_mixture_model.py)
    motif_file = cfg["input_data_parquet"]
    clipseq_file = cfg["clipseq_bed"]
    pipseq_file = cfg["pipseq_parquet"]
    rbp_name = cfg.get("rbp_name", "RBP")
    motif_id = cfg.get("motif_id", "Motif")
    motif_logo = cfg.get("motif_logo", None)
    output_root = Path(cfg.get("output_root", "results"))
    score_phastcons100_threshold = float(cfg.get("score_phastcons100_threshold", 1.0))
    motif_range = int(cfg.get("motif_range", 50))
    model_json_path = cfg.get("model_json")

    if model_json_path is None:
        raise SystemExit("Config must provide 'model_json' path to a trained model JSON produced by 'accmix model'.")

    output_root.mkdir(parents=True, exist_ok=True)

    file_title = f"{rbp_name}_{motif_id}"

    motif_df = pl.read_parquet(motif_file)

    # Read CLIP-seq with pandas (more robust to odd encodings), then convert to Polars
    clipseq_pd = pd.read_csv(
        clipseq_file,
        sep="\t",
        header=None,
        names=[
            "Chromosome",
            "Start",
            "End",
            "peak_id",
            "score",
            "Strand",
            "RBP_Name",
            "method",
            "cell_line",
            "datasource",
        ],
    )
    clipseq_df = pl.from_pandas(clipseq_pd)

    motif_df = motif_df.with_columns([
        pl.col("id").str.split("_").alias("id_split")
    ]).with_columns([
        pl.col("id_split").list.get(0).alias("Chromosome"),
        pl.col("id_split").list.get(2).alias("Strand"),
        pl.col("id_split").list.get(1).cast(pl.Int64).alias("Position"),
    ]).drop("id_split")

    motif_df = motif_df.with_columns([
        (pl.col("Position") - motif_range).alias("Start"),
        (pl.col("Position") + motif_range).alias("End"),
    ])

    motif_pr = pr.PyRanges(motif_df.to_pandas())
    clipseq_pr = pr.PyRanges(
        clipseq_df.select([
            "Chromosome",
            "Start",
            "End",
            "Strand",
            "peak_id",
            "score",
        ]).to_pandas()
    )

    pipseq_pr = pr.PyRanges(pl.read_parquet(pipseq_file).to_pandas())

    # pyranges >= 0.3 uses `overlap` instead of `intersect`
    intersections = motif_pr.overlap(clipseq_pr, strand_behavior="same")
    pipintersections = motif_pr.overlap(pipseq_pr, strand_behavior="same")

    intersected_ids = set(intersections["id"].tolist()) if len(intersections) > 0 else set()
    pipintersected_ids = set(pipintersections["id"].tolist()) if len(pipintersections) > 0 else set()

    motif_df = motif_df.with_columns([
        pl.when(pl.col("id").is_in(list(intersected_ids)))
        .then(pl.lit("clip_bound"))
        .when(
            ~pl.col("id").is_in(list(pipintersected_ids))
            & (pl.col("score_phastcons100") <= score_phastcons100_threshold)
        )
        .then(pl.lit("clip_unbound"))
        .otherwise(pl.lit("non_determined"))
        .alias("source"),
    ])

    motif_df = motif_df.drop(["Chromosome", "Position", "Start", "End"])

    feature_columns: List[str] = [
        "inner_mean_logPWM",
        "outer_mean_logPWM",
        "GC_inner_pct",
        "GC_outer_pct",
        "TSS_proximity",
        "PhastCons100_percent",
        "TPM",
        "score_phastcons100",
        "score_phylop100",
    ]
    skip_normalization = [
        "TSS_proximity",
        "PhastCons100_percent",
        "score_phastcons100",
    ]

    df, s, X, site_ids = read_polars_input(motif_df, "id", "s_l", feature_columns)
    X_zscore = X.copy()
    for i, col in enumerate(feature_columns):
        if col not in skip_normalization:
            col_zscore = stats.zscore(X[:, i])
            X_zscore[:, i] = np.nan_to_num(col_zscore, nan=0.0)
    X_zscore[:, 0] = 1

    # Load trained model parameters and recompute prior_p/posterior_r
    model = json.loads(Path(model_json_path).read_text())
    beta = np.asarray(model["beta"], dtype=float)
    mu0 = float(model["gaussian_params"]["mu0"])
    sigma0 = float(model["gaussian_params"]["sigma0"])
    mu1 = float(model["gaussian_params"]["mu1"])
    sigma1 = float(model["gaussian_params"]["sigma1"])

    log_s = np.log(s + 1)
    eta = X_zscore @ beta
    p = 1.0 / (1.0 + np.exp(-eta))
    eps = 1e-10
    p_bg = stats.norm.pdf(log_s, mu0, sigma0 + eps)
    p_sig = stats.norm.pdf(log_s, mu1, sigma1 + eps)
    denominator = (1 - p) * p_bg + p * p_sig + eps
    r = (p * p_sig) / denominator

    out_df = df.with_columns([
        pl.Series("prior_p", p),
        pl.Series("posterior_r", r),
    ])
    out_df = utils.change_label(out_df)

    roc_auc = utils.plot_em_auc(out_df, file_title, save_path=None)
    roc_auc_control = utils.plot_em_auc_control(out_df)

    expanded_feature_columns = feature_columns + [
        "prior_p",
        "posterior_r",
        "s_l",
        "source",
    ]

    plots_dir = output_root / "plots"
    logs_dir = output_root / "logs"
    plots_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    utils.heatmap_plot(
        data=out_df,
        columns=expanded_feature_columns,
        no_normalize_columns=skip_normalization + ["source"],
        save_heatmap_path=str(plots_dir / f"{file_title}_heatmap_with_title_logo.png"),
        file_title=file_title,
        motif_logo=motif_logo,
    )

    dist_dir = plots_dir / "distribution"
    dist_dir.mkdir(parents=True, exist_ok=True)
    test_dict = utils.compute_qq_validate_distribution(
        out_df,
        save_figure_path=str(dist_dir / f"{file_title}.dist.val.png"),
    )

    source_mask = out_df["source"] != "non_determined"
    X_source = X_zscore[source_mask.to_numpy(), 1:]
    y_source_labels = out_df.filter(source_mask)["source"].to_numpy()
    y_source = (y_source_labels == "clip_bound").astype(int)

    lr_source = LogisticRegression(max_iter=1000, random_state=42)
    lr_source.fit(X_source, y_source)
    y_source_proba = lr_source.predict_proba(X_source)[:, 1]
    fpr_source, tpr_source, _ = roc_curve(y_source, y_source_proba)
    auc_source = auc(fpr_source, tpr_source)

    X_zscore_s_l = X_zscore.copy()
    X_zscore_s_l[:, 0] = stats.zscore(log_s)
    X_source_s_l = X_zscore_s_l[source_mask.to_numpy(), :]
    lr_source_s_l = LogisticRegression(max_iter=1000, random_state=42)
    lr_source_s_l.fit(X_source_s_l, y_source)
    y_source_proba_s_l = lr_source_s_l.predict_proba(X_source_s_l)[:, 1]
    fpr_source_s_l, tpr_source_s_l, _ = roc_curve(y_source, y_source_proba_s_l)
    auc_source_s_l = auc(fpr_source_s_l, tpr_source_s_l)

    beta_results_df_s_l, beta_results_df, metadata_df = utils.save_analysis_results(
        model,
        lr_source.coef_[0],
        lr_source_s_l.coef_[0],
        lr_source.intercept_[0],
        lr_source_s_l.intercept_[0],
        feature_columns,
        roc_auc,
        roc_auc_control,
        auc_source,
        auc_source_s_l,
        file_title,
        out_df,
        test_dict=test_dict,
        motif_df=motif_df,
        output_dir=str(logs_dir),
    )

    print("Evaluation completed for", file_title)
