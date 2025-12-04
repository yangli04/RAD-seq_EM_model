"""Batch driver to run accmix models over ../midway3_results/*.parquet.

This mirrors the old run_mixture_model.py behavior but delegates all
model fitting and evaluation to the installed accmix package.
"""

from __future__ import annotations

import glob
import os
from pathlib import Path
from typing import Dict

import polars as pl

from accmix.model_core import fit_model


SCORE_PHASTCONS100_THRESHOLD: float = 1.0
MOTIF_RANGE: int = 50


def _build_rbp_table(midway_results_dir: str, rbp_info_path: str, clipseq_glob: str) -> pl.DataFrame:
    """Recreate the mapping table from motif parquet files to RBP/CLIP files.

    This intentionally mirrors the logic in scripts/run_mixture_model.py but
    keeps it inside this helper so the main script remains small.
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

    midway_results_dir = str(base_dir / "data"/  "midway3_results")
    rbp_info_path = str(base_dir / "data" / "RBP_Information_all_motifs.txt")
    clipseq_glob = "/home/yangli/train/add_rbp_clip/raw_data/*_HeLa.bed"

    rbp_df = _build_rbp_table(midway_results_dir, rbp_info_path, clipseq_glob)

    for idx in range(len(rbp_df)):
        try:
            motif_file = rbp_df["motif_file"][idx]
            rbp_name = rbp_df["RBP_Name"][idx]
            motif_id = rbp_df["Motif_ID"][idx]

            file_title = f"{rbp_name}_{motif_id}"
            print(f"[accmix batch] Fitting model for {file_title} from {motif_file}")

            # This assumes each motif parquet is already in the annotated format
            # expected by accmix.model_core.fit_model (same as accmix model -i).
            out_prefix = str(base_dir / "results" / file_title)

            fit_model(
                input_parquet=motif_file,
                out_prefix=out_prefix,
                rbp_name=rbp_name,
                motif_id=motif_id,
            )
        except Exception as exc:  # keep going on individual failures
            print(f"[accmix batch] ERROR for index {idx}: {exc}")


if __name__ == "__main__":
    main()
