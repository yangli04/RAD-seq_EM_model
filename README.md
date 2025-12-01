# EM model for RAD-seq

This repository implements a site-level mixture model that scores genomic motif sites using a depth-weighted RAD/AS signal (`s_l`) and a logistic prior built from sequence and annotation features.

- **Model**: two-component mixture (background vs signal) on site score `s_l`; prior p_l = sigmoid(X beta). EM / MLE / hybrid estimation available (see `scripts/model.py`). Gaussian variant implemented as `run_em_Gaussian`.
- **Key scripts**:
  - `scripts/scan_pwm_regions.py`: genome-wide PWM & GC scanner -> motif region scores (`*_topA.tsv.gz` / `*_botB.tsv.gz`).
  - `scripts/make_table.py`: computes site statistic `s_l` from accessibility (AS native/fixed) tables and PWM priors -> parquet with `s_l`.
  - `scripts/annotate_TSS.py`: maps sites to transcripts, adds TSS proximity, conservation, TPM, and other annotations.
  - `scripts/model.py`: core model and estimators (modes: `empirical`, `em`, `mle`, `hybrid`); cli produces `<out>.parquet` (adds `prior_p`, `posterior_r`) and `<out>.json` (params).
  - `scripts/run_mixture_model.py`: high-level driver that loads motif tables, runs EM/Gaussian EM, generates plots and saves results.
  - `scripts/em_utils.py`: plotting, diagnostics, and result-saving helpers.
  - `scripts/compute_diff_bind.py`: compares WT vs KD results and writes differential-binding tables using posterior thresholds.

- **Input format**: TSV/Parquet with columns: unique site id (default `site`/`id`), `s_l` (precomputed), and feature columns (examples: `inner_mean_logPWM`, `outer_mean_logPWM`, `GC_inner_pct`, `GC_outer_pct`, `TSS_proximity`, `PhastCons100_percent`, `TPM`, `score_phastcons100`, `score_phylop100`). Use `scripts/make_table.py` + `annotate_TSS.py` to prepare inputs.

- **Outputs**: `<out>.parquet` (input plus `prior_p`, `posterior_r`), `<out>.json` (fitted params); analysis and evaluation scripts also write plots and TSVs under `../logs`, `../plots`, and `../predicted_bound`.

- **Quick example**:
  - Prepare input TSV/parquet with `s_l` and features.
  - Run model (hybrid mode recommended):

```bash
python3 scripts/model.py --tsv input.tsv --feat-cols inner_mean_logPWM,outer_mean_logPWM,GC_inner_pct,GC_outer_pct,TSS_proximity,PhastCons100_percent,TPM,score_phastcons100,score_phylop100 --out-prefix results/motif1 --mode hybrid
```

- **Notes**: features are typically z-scored before fitting (see `scripts/run_mixture_model.py`); posterior thresholds used in downstream scripts: e.g. `posterior_r > 0.99` => bound, `<0.5` => unbound, others uncertain. See `scripts/compute_diff_bind.py` for differential-binding logic.

For details, inspect the docstrings in `scripts/model.py` and helpers in `scripts/em_utils.py`.