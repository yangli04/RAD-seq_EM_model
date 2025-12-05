---
title: Reproducibility
---

# Reproducibility

This guide shows how to reproduce the results included in this repository, using the standard workflow: scan PWM → annotate AS → annotate TSS → fit model → evaluate.

## Prerequisites

- Install the package (PyPI or source). See [Installation](installation.md).
- Prepare data as described in [Data](data.md). Example paths below assume the `data/` layout in this repo.

## 1) Scan PWM

```bash
accmix scan \
  -f data/fasta/test.fa \
  -p data/pwms/M00124_example.txt \
  -o results/M00124_example
```

Produces: `results/M00124_example_topA.tsv.gz`, `results/M00124_example_botB.tsv.gz`.

## 2) Annotate AS (compute s_l)

```bash
accmix annotate-acc \
  -n data/AS/ANC1C.hisat3n_table.bed6 \
  -f data/AS/ANC1xC.hisat3n_table.bed6 \
  -t results/M00124_example_topA.tsv.gz \
  -o results/M00124_example_sl.parquet \
  --M 50 --N 500
```

## 3) Annotate TSS / conservation / TPM

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

## 4) Fit EM model

```bash
accmix model \
  -i results/M00124_example_annotated.parquet \
  -o results/RBP_Motif \
  -r ExampleRBP \
  -m M00124
```

## 5) Evaluate

Create a config JSON (see [CLI](cli.md#evaluate-model)) and run:

```bash
accmix evaluate -c results/ExampleRBP_M00124.evaluate.config.json
```

## Embedded results

Below are sample outputs generated from the pipeline. These images link directly to the repository artifacts.

### Heatmap

![Example heatmap](assets/figures/ExampleRBP_M00124_heatmap_with_title_logo.png)

### Distribution

![Example distribution](assets/figures/ExampleRBP_M00124.dist.val.png)

> Tip: For your own runs, you’ll find similar outputs under `results/plots/` with `{RBP}_{MotifID}*.png`.
