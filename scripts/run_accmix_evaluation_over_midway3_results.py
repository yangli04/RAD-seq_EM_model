"""Batch driver to run accmix evaluation over models from midway3_results.

This script mirrors the RBP/motif mapping from run_mixture_model.py and the
model batch driver, but delegates evaluation to accmix.evaluate.run_evaluation.

It expects that models have already been fitted by accmix (e.g. using the
run_accmix_models_over_midway3_results.py script), producing model JSON files
and (optionally) model parquets under ``results/``.
"""

from __future__ import annotations

import glob
import json
import os
import subprocess
from pathlib import Path
from typing import Dict

import polars as pl


def _build_rbp_table(midway_results_dir: str, rbp_info_path: str, clipseq_glob: str) -> pl.DataFrame:
    """Recreate the mapping table from motif parquet files to RBP/CLIP files.

    Shared with the modeling batch script so that evaluation uses the same
    RBP/motif pairs.
    """

    motif_files = sorted(glob.glob(os.path.join(midway_results_dir, "M*_wtTSS_and*.parquet")))
    motif_selected = [
        os.path.basename(f).replace("_wtTSS_and_PhaseCons.parquet", "")
        for f in motif_files
    ]

    clipseq_files = sorted(glob.glob(clipseq_glob))
    clipseq_selected = [
        os.path.basename(i).replace("_HeLa.bed", "") for i in clipseq_files
    ]

    rbp_df = pl.read_csv(rbp_info_path, separator="\t").select(["Motif_ID", "RBP_Name"])

    motif_map: Dict[str, str] = {m: f for m, f in zip(motif_selected, motif_files)}
    clipseq_map: Dict[str, str] = {c: f for c, f in zip(clipseq_selected, clipseq_files)}

    rbp_df = rbp_df.filter(
        pl.col("Motif_ID").is_in(motif_selected)
        & pl.col("RBP_Name").is_in(clipseq_selected)
    ).with_columns(
        [
            pl.col("Motif_ID").map_elements(motif_map.get, return_dtype=str).alias("motif_file"),
            pl.col("RBP_Name").map_elements(clipseq_map.get, return_dtype=str).alias("clipseq_file"),
        ]
    ).with_columns(
        (
            pl.concat_str(
                pl.lit(
                    "/home/yangli/workspace/accessibility/CISBP_RNA/logos_all_motifs/"
                ),
                pl.col("Motif_ID"),
                pl.lit("_fwd.png"),
            )
        ).alias("motif_logo")
    )

    return rbp_df


def main() -> None:
    base_dir = Path(__file__).resolve().parent.parent

    midway_results_dir = str(base_dir / "data" / "midway3_results")
    rbp_info_path = str(base_dir / "data" / "RBP_Information_all_motifs.txt")
    clipseq_glob = "/home/yangli/train/add_rbp_clip/raw_data/*_HeLa.bed"

    clipseq_bed_default = str(base_dir / "data" / "clipseq" / "ELAVL1_HeLa.bed")
    pipseq_parquet = str(base_dir / "data" / "evaluation" / "PIPseq_HeLa.parquet")

    rbp_df = _build_rbp_table(midway_results_dir, rbp_info_path, clipseq_glob)
    for idx in range(len(rbp_df)):
        try:
            motif_file = rbp_df["motif_file"][idx]
            rbp_name = rbp_df["RBP_Name"][idx]
            motif_id = rbp_df["Motif_ID"][idx]
            motif_logo = rbp_df["motif_logo"][idx]

            file_title = f"{rbp_name}_{motif_id}"
            print(f"[accmix batch eval] Evaluating model for {file_title}")

            # Model JSON produced by `accmix model` / accmix.model_core.fit_model. 
            model_json_candidates = sorted(
                glob.glob(str(base_dir / "results" / f"{file_title}*.model.json"))
            )
            if not model_json_candidates:
                print(f"[accmix batch eval] No model JSON found for {file_title}, skipping.")
                continue
            # Get newest model here. 
            model_json = model_json_candidates[-1]

            # Input parquet should be the same file that was used for fitting.
            input_data_parquet = motif_file

            # CLIP bed – use the per-RBP mapping if available, otherwise fall back.
            clipseq_bed = (
                rbp_df["clipseq_file"][idx]
                if "clipseq_file" in rbp_df.columns
                else clipseq_bed_default
            )

            cfg = {
                "input_data_parquet": input_data_parquet,
                "clipseq_bed": clipseq_bed,
                "pipseq_parquet": pipseq_parquet,
                "rbp_name": rbp_name,
                "motif_id": motif_id,
                "motif_logo": motif_logo,
                "output_root": str(base_dir / "results"),
                "model_json": model_json,
                "score_phastcons100_threshold": 1.0,
                "motif_range": 50,
            }

            # Write a temp config file and call the accmix CLI.
            cfg_path = base_dir / "results" / f"{file_title}.evaluate.config.json"
            cfg_path.write_text(json.dumps(cfg, indent=2))

            subprocess.run(
                ["accmix", "evaluate", "-c", str(cfg_path)],
                check=True,
            )
        except Exception as exc:  # keep going on individual failures
            print(f"[accmix batch eval] ERROR for index {idx}: {exc}")


if __name__ == "__main__":
    main()
