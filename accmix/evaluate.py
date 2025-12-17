from __future__ import annotations
from pathlib import Path
from typing import List, Optional

import numpy as np
import pyranges as pr
import polars as pl
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc

from accmix.model_core import read_polars_input
from accmix import utils as utils


def run_evaluation(
    model_parquet: str,
    clipseq_bed: str,
    pipseq_parquet: str,
    rbp_name: str,
    motif_id: str,
    output_root: str,
    motif_logo: Optional[str] = None,
    score_phastcons100_threshold: float = 1.0,
    motif_range: int = 50,
    name_add: str = "",
) -> None:
    """Run evaluation pipeline.

    Parameters
    - model_parquet: Parquet produced by `accmix model` containing prior/posterior.
    - clipseq_bed: CLIP-seq peaks BED path.
    - pipseq_parquet: PIP-seq parquet path.
    - rbp_name: RBP label for titles/filenames.
    - motif_id: Motif identifier for titles/filenames.
    - output_root: Directory to write plots and logs.
    - motif_logo: Optional logo to display on heatmaps.
    - score_phastcons100_threshold: Threshold for class labeling logic.
    - motif_range: Window half-size around site position for overlap checks.
    - name_add: Additional string to append to generated filenames.
    """

    output_root = Path(output_root)

    output_root.mkdir(parents=True, exist_ok=True)

    file_title = f"{rbp_name}_{motif_id}_{name_add}" if name_add else f"{rbp_name}_{motif_id}"

    df = pl.read_parquet(model_parquet)
    missing_cols = [c for c in ("prior_p", "posterior_r", "id", "s_l") if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Model parquet missing required columns: {missing_cols}")

    # Read CLIP-seq with pandas (more robust to odd encodings), then convert to Polars
    clipseq_pd = pd.read_csv(
        clipseq_bed,
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

    df = df.with_columns([
        pl.col("id").str.split("_").alias("id_split")
    ]).with_columns([
        pl.col("id_split").list.get(0).alias("Chromosome"),
        pl.col("id_split").list.get(2).alias("Strand"),
        pl.col("id_split").list.get(1).cast(pl.Int64).alias("Position"),
    ]).drop("id_split")

    df = df.with_columns([
        (pl.col("Position") - motif_range).alias("Start"),
        (pl.col("Position") + motif_range).alias("End"),
    ])

    motif_pr = pr.PyRanges(df.to_pandas())
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

    pipseq_pr = pr.PyRanges(pl.read_parquet(pipseq_parquet).to_pandas())

    # pyranges >= 0.3 uses `overlap` instead of `intersect`
    intersections = motif_pr.overlap(clipseq_pr, strand_behavior="same")
    pipintersections = motif_pr.overlap(pipseq_pr, strand_behavior="same")

    intersected_ids = set(intersections["id"].tolist()) if len(intersections) > 0 else set()
    pipintersected_ids = set(pipintersections["id"].tolist()) if len(pipintersections) > 0 else set()

    df = df.with_columns([
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

    df = df.drop(["Chromosome", "Position", "Start", "End"])

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

    df_pl, s, X, site_ids = read_polars_input(df, "id", "s_l", feature_columns)
    X_zscore = X.copy()
    for i, col in enumerate(feature_columns):
        if col not in skip_normalization:
            col_zscore = stats.zscore(X[:, i])
            X_zscore[:, i] = np.nan_to_num(col_zscore, nan=0.0)
    X_zscore[:, 0] = 1
    log_s = np.log(s + 1)

    out_df = df_pl
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
        None,
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
        motif_df=df,
        output_dir=str(logs_dir),
    )

    print("Evaluation completed for", file_title)
