---
title: Tutorial
---

# Tutorial

This tutorial walks through the full workflow: scan PWM → annotate accessibility (AS) → annotate TSS/conservation/TPM → fit EM model → evaluation.

## 1) Scan PWM and GC

Scan the genome FASTA with a PWM and compute GC content per region.

```bash
accmix scan \
  -f data/fasta/test.fa \
  -p data/pwms/M00124_example.txt \
  -o results/M00124_example
```

Outputs:

- `results/M00124_example_topA.tsv.gz`
- `results/M00124_example_botB.tsv.gz`

## 2) Annotate accessibility (compute s_l)

Compute accessibility-derived site score `s_l` using native vs. fixed alignments and PWM toptables.

```bash
accmix annotate-acc \
  -n data/AS/ANC1C.hisat3n_table.bed6 \
  -f data/AS/ANC1xC.hisat3n_table.bed6 \
  -t results/M00124_example_topA.tsv.gz \
  -o results/M00124_example_sl.parquet \
  --M 50 \
  --N 500
```

Notes:

- `--M` is the inner flank; `--N` is the outer flank; outer flank length = `N - M`.

## 3) Annotate TSS / conservation / TPM

Join transcript/TSS proximity, conservation (phastCons/phyloP), and RNA-seq TPM to each site.

```bash
accmix annotate-tss \
  -i results/M00124_example_sl.parquet \
  -o results/M00124_example_annotated.parquet \
  -r data/evaluation/RNAseq_HeLa_TPM.parquet \
  -c data/evaluation/phastCons100way.bed.gz \
  -p data/evaluation/phastCons100way.parquet \
  -y data/evaluation/phyloP100way.parquet \
  -R data/fasta/test.fa
```

## 4) Fit EM model (Gaussian + logistic prior)

Fit the mixture model and write posterior probabilities and model parameters.

```bash
accmix model \
  -i results/M00124_example_annotated.parquet \
  -o results/RBP_Motif \
  -r ExampleRBP \
  -m M00124
```

Outputs:

- `results/RBP_Motif.XXXXXX.model.parquet` (data with `prior_p`, `posterior_r`)
- `results/RBP_Motif.XXXXXX.model.json` (fitted parameters)

## 5) Evaluation

Evaluate the fitted model against external assays (e.g., CLIP/PIPseq) via the `evaluate` subcommand using a config file.

Example config (`results/ExampleRBP_M00124.evaluate.config.json`):

```json
{
  "input_data_parquet": "results/M00124_example_annotated.parquet",
  "clipseq_bed": "data/clipseq/ELAVL1_HeLa.bed",
  "pipseq_parquet": "data/evaluation/PIPseq_HeLa.parquet",
  "rbp_name": "ExampleRBP",
  "motif_id": "M00124",
  "motif_logo": "data/logos/M00124_fwd.png",
  "output_root": "results",
  "model_json": "results/RBP_Motif.XXXXXX.model.json",
  "score_phastcons100_threshold": 1.0,
  "motif_range": 50
}
```

Run evaluation:

```bash
accmix evaluate -c results/ExampleRBP_M00124.evaluate.config.json
```

This writes plots/tables under `results/` for downstream analysis.
