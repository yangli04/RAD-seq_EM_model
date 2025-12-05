<p align="center">
  <img src="assets/accmix-logo.svg" alt="accmix logo" width="360" />
  <br>
  <sub>Under active development — interfaces may evolve</sub>
  <br/>
  <a href="https://github.com/yangli04/RAD-seq_EM_model/blob/main/LICENSE" target="_blank">
    <img src="https://img.shields.io/badge/license-CC%20BY--NC--ND%204.0-blue" alt="License: CC BY-NC-ND 4.0" />
  </a>
  <a href="https://pypi.org/project/accmix/" target="_blank">
    <img src="https://img.shields.io/pypi/v/accmix.svg" alt="PyPI version" />
  </a>
</p>

# accmix: Accessibility Mixture Model

CLI toolkit to (1) scan PWMs genome-wide, (2) compute accessibility-derived site scores `s_l`, (3) annotate with TSS/conservation/TPM, and (4) fit and evaluate a Gaussian EM with a logistic prior.

## Installation

Install from github repository
```bash
pip install -e .
```

Install from pypi
```bash
pip install accmix
```

## Documentation

Full user and API documentation is available at:

- https://yangli04.github.io/RAD-seq_EM_model/

## Data layout

Example inputs are referenced under `data/...`:

- `data/fasta/test.fa` – small test genome FASTA.
- `data/pwms/M00124_example.txt` – example PWM for scanning.
- `data/clipseq/ELAVL1_HeLa.bed` – example CLIP-seq peaks.


## CLI overview

The main entry point is `accmix` with subcommands:

### 1. Scan PWM and GC

```bash
accmix scan \
  -f data/fasta/test.fa \
  -p data/pwms/M00124_example.txt \
  -o results/M00124_example
```

Outputs (for example PWM):

- `results/M00124_example_topA.tsv.gz`
- `results/M00124_example_botB.tsv.gz`

### 2. Annotate accessibility (compute `s_l`)

```bash
accmix annotate-acc \
  -n data/AS/ANC1C.hisat3n_table.bed6 \
  -f data/AS/ANC1xC.hisat3n_table.bed6 \
  -t results/M00124_example_topA.tsv.gz \
  -o results/M00124_example_sl.parquet
```

Important options:

- `-M / --M` – inner flank size (default: 50).
- `-N / --N` – outer flank size (default: 500). Outer flank length is `N - M`.

### 3. Annotate TSS / conservation / TPM

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

### 4. Fit EM model (Gaussian + logistic prior)

```bash
accmix model \
  -i results/M00124_example_annotated.parquet \
  -o results/RBP_Motif \
  -r ExampleRBP \
  -m M00124
```

Outputs:

  - `results/RBP_Motif.XXXXXX.model.parquet` – input data with `prior_p` and `posterior_r`.
  - `results/RBP_Motif.XXXXXX.model.json` – fitted model parameters.

## Notes

- Dependencies: `polars`, `pyranges`, `metagene`, `numpy`, `scipy`, `pyarrow`, `numba`, `tqdm`, `typer[all]`, `scikit-learn`, `matplotlib`, `seaborn`, `pandas`.
- The CLI options mirror the underlying scripts; run `accmix <command> --help` for full details.


All files under `data/` can be downloaded from:

https://(university of chicago's short name, with 8 letters).box.com/s/(remove following first nine letters and final 6 letters)meipiannitsmywbgimvl47w9nynq66hur3c8dnirvzhende

## If you use this package, please cite:

Li Y., *accmix: Accessibility Mixture Model*, GitHub repository, https://github.com/yangli04/RAD-seq_EM_model