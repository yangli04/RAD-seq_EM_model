---
title: API Reference
---

# API Reference

This page documents the primary user-facing interfaces (CLI) and the Python modules. For Python, we auto-generate reference docs via mkdocstrings.

## CLI subcommands

### `accmix scan`

- `-f, --fasta` (str): Genome FASTA path.
- `-p, --pwm` (str): PWM file (CISBP-like format).
- `-o, --out-prefix` (str): Output prefix for TSV.gz results.

Outputs: `<prefix>_topA.tsv.gz`, `<prefix>_botB.tsv.gz`.

### `accmix annotate-acc`

- `-n, --ASnative` (str): Native alignment BED6 table.
- `-f, --ASfixed` (str): Fixed alignment BED6 table.
- `-t, --toptable` (str): PWM toptable (e.g., from `scan`).
- `-o, --out-parquet` (str): Output parquet with `s_l`.
- `--M` (int, default 50): Inner flank size.
- `--N` (int, default 500): Outer flank size; outer flank length is `N - M`.

### `accmix annotate-tss`

- `-i, --input-parquet` (str): Input parquet from `annotate-acc`.
- `-o, --out-parquet` (str): Output parquet with annotations.
- `-r, --rna-seq-parquet` (str): RNA-seq TPM parquet.
- `-c, --phastcons-bed` (str): phastCons bed.gz track.
- `-p, --phastcons-parquet` (str): phastCons parquet track.
- `-y, --phylop-parquet` (str): phyloP parquet track.
- `-R, --reference` (str): Reference genome FASTA.

### `accmix model`

- `-i, --input` (str): Annotated parquet containing features.
- `-o, --out-prefix` (str): Output prefix for parquet+JSON artifacts.
- `-r, --rbp-name` (str): RBP label used in outputs.
- `-m, --motif-id` (str): Motif identifier.
- `--feat-cols` (list[str], optional): Feature columns to model.
- `--site-col` (str, optional): Column indicating site/grouping.
- `--s-col` (str, optional): `s_l` column name.

Outputs: `<prefix>*.model.parquet`, `<prefix>*.model.json`.

### `accmix evaluate`

Uses a JSON config via `-c/--config` with keys:

- `input_data_parquet` (str)
- `clipseq_bed` (str)
- `pipseq_parquet` (str, optional)
- `rbp_name` (str)
- `motif_id` (str)
- `motif_logo` (str, optional)
- `output_root` (str)
- `model_json` (str)
- `score_phastcons100_threshold` (float)
- `motif_range` (int)

## Batch helper scripts

### `scripts/run_accmix_models_over_midway3_results.py`

Helper to batch-fit models across many annotated motif parquet files.

Functions:

- `_build_rbp_table(midway_results_dir: str, rbp_info_path: str, clipseq_glob: str) -> polars.DataFrame`
  - Purpose: Map motif IDs and RBP names to file paths, producing columns `motif_file`, `clipseq_file`, `motif_logo`.
  - Params:
    - `midway_results_dir`: Directory containing annotated motif parquet files (pattern `M*_wtTSS_and*.parquet`).
    - `rbp_info_path`: TSV with columns `Motif_ID`, `RBP_Name`.
    - `clipseq_glob`: Glob pattern to resolve CLIP-seq BED files (e.g., `*_HeLa.bed`).
  - Returns: DataFrame with selected pairs for batch processing.

- `main() -> None`
  - Iterates over the mapping table and calls `accmix.model_core.fit_model(...)` on each file, writing outputs under `results/`.

### `scripts/run_accmix_evaluation_over_midway3_results.py`

Helper to batch-evaluate previously fitted models.

Functions:

- `_build_rbp_table(midway_results_dir: str, rbp_info_path: str, clipseq_glob: str) -> polars.DataFrame`
  - Same semantics as above to ensure consistent RBP/motif pairs for evaluation.

- `main() -> None`
  - For each RBP/motif pair, locates the latest model JSON under `results/`, prepares a config file, and invokes `accmix evaluate -c <config.json>`.

## Python modules

### `accmix.scanner`

::: accmix.scanner

### `accmix.annotate`

::: accmix.annotate

### `accmix.model_core`

::: accmix.model_core

### `accmix.evaluate`

::: accmix.evaluate

### `accmix.utils`

::: accmix.utils

## Notes

- The CLI is the stable user-facing API; Python internals may evolve.
- For automated API docs in the future, we can add `mkdocstrings` once the package structure is finalized.
