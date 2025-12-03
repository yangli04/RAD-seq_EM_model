# accmix: Accessibility Mixture Model

<img src="assets/accmix-logo.svg" alt="accmix logo" width="420">

Concise CLI package to: (1) scan PWM genome-wide, (2) compute accessibility-derived site score `s_l`, (3) annotate with TSS/conservation/TPM, and (4) fit a Gaussian EM with a logistic prior.

Status: under active development. Interfaces may evolve; please pin commits for reproducibility.

Installation

```bash
pip install -e .
```

Commands

- Scan PWM and GC
```bash
accmix scan \
  --fasta genome.fa.gz \
  --pwm motif.txt \
  --out-prefix results/M00124
```
Outputs: `results/M00124_topA.tsv.gz`, `results/M00124_botB.tsv.gz`.

- Annotate accessibility (compute `s_l`)
```bash
accmix annotate-acc \
  --ASnative path/ANC1C.hisat3n_table.bed6 \
  --ASfixed  path/ANC1xC.hisat3n_table.bed6 \
  --toptable results/M00124_topA.tsv.gz \
  --out results/top_sl_table.parquet
```
Uses inner `M=50`, outer `N=500` (defaults); outer flank length is `N-M`.

- Annotate TSS/conservation/TPM
```bash
accmix annotate-tss \
  --input-parquet results/top_sl_table.parquet \
  --out results/annotated.parquet
```

- Fit EM model (Gaussian + logistic prior)
```bash
accmix model \
  --input-parquet results/annotated.parquet \
  --out-prefix results/RBP_Motif
```
Outputs: `<out>.model.parquet` with `prior_p` and `posterior_r`, and `<out>.model.json` with parameters.

Notes

- Dependencies: `polars`, `pyranges`, `metagene`, `numpy`, `scipy`, `pyarrow`, `numba`, `tqdm` (installed via `pip install -e .`).
- The CLI exposes key flags from the original scripts to maintain identical behavior.