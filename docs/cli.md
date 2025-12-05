---
title: CLI
---

# CLI

The main entry point is `accmix` with these subcommands. Run `accmix <cmd> --help` for the authoritative list. Below are the parameters, defaults, and meanings.

## `accmix scan`

Scan a genome FASTA with a PWM and compute inner/outer scores.

```bash
accmix scan -f <fasta> -p <pwm.txt> [-o <out_prefix>] [options]
```

Options

- `-f, --fasta PATH` (required): Genome FASTA (.fa/.fa.gz).
- `-p, --pwm PATH` (required): PWM text file with columns Pos,A,C,G,T/U.
- `-o, --out-prefix PATH`: Output prefix (default: `results/<pwm>.<timestamp>`).
- `-M INT` (default 50): Inner half-width in bp.
- `-N INT` (default 500): Outer half-width in bp; must be > `M`.
- `-A INT` (default 2_000_000): Keep top-A entries per strand.
- `-B INT` (default 2_000_000): Keep bottom-B entries per strand.
- `-g, --bg "A,C,G,T"`: Background base probabilities for LLR (overridden by `-G`).
- `-G, --bg-from-fasta`: Estimate background from the provided FASTA.

Outputs: `results/<prefix>_topA.tsv.gz`, `results/<prefix>_botB.tsv.gz`.

## `accmix annotate-acc`

Compute accessibility-derived `s_l` from AS native/fixed and PWM top table.

```bash
accmix annotate-acc -n <ASnative> -f <ASfixed> -t <toptable> [-o <out>] [options]
```

Options

- `-n, --ASnative PATH` (required): AS native file (chr, position, strand, score, depth, motif info).
- `-f, --ASfixed PATH` (required): AS fixed BED6 (chr, start, end, strand, score, depth).
- `-t, --toptable PATH` (required): PWM scan top-table TSV.gz with inner/outer scores.
- `-o, --out PATH`: Output parquet (default: `results/sl.<timestamp>.parquet`).
- `-M INT` (default 50): Inner half-width used in scanning.
- `-N INT` (default 500): Outer half-width used in scanning.

Output: parquet with `s_l` per site.

## `accmix annotate-tss`

Add TSS proximity, conservation, and TPM annotations to sites.

```bash
accmix annotate-tss -i <input.parquet> [-o <out>] [options]
```

Options

- `-i, --input-parquet PATH` (required): Input parquet (e.g., output of `annotate-acc`).
- `-o, --out PATH`: Output parquet (default: `results/annotated.<timestamp>.parquet`).
- `-r, --rna-seq-parquet PATH` (default `data/expression/ENCFF364YCB_HeLa_RNAseq_Transcripts_count_curated.parquet`): Transcript TPM parquet.
- `-c, --phastcons-bed PATH` (default `data/conservation_score/phastCons100way1.bed`): phastCons BED for regional conservation.
- `-p, --phastcons-parquet PATH` (default `data/conservation_score/hg38.phastCons100way.1.parquet`): Base-level phastCons scores.
- `-y, --phylop-parquet PATH` (default `data/conservation_score/hg38.phyloP100way.1.parquet`): Base-level phyloP scores.
- `-R, --reference TEXT` (default `GRCh38`): Reference genome name.

Output: parquet with added TSS distance, conservation, and expression features.

## `accmix model`

Fit Gaussian EM with logistic prior and write priors/posteriors plus model JSON.

```bash
accmix model -i <annotated.parquet> -o <out_prefix> -r <RBP> -m <MotifID> [options]
```

Options

- `-i, --input-parquet PATH` (required): Annotated parquet with `id`, `s_l`, and features.
- `-o, --out-prefix PATH`: Output prefix (default: `results/<rbp>_<motif>.<timestamp>`).
- `-r, --rbp-name TEXT` (default `RBP`): RBP label for outputs.
- `-m, --motif-id TEXT` (default `Motif`): Motif ID label for outputs.
- `-F, --feat-cols TEXT`: Comma-separated feature names (default: built-in set).
- `-s, --site-col TEXT` (default `id`): Site identifier column.
- `-S, --s-col TEXT` (default `s_l`): Accessibility score column.
- `-M, --mode TEXT` (default `em`): Modeling mode (currently only `em`).

Outputs: `<prefix>.model.parquet` (priors/posteriors) and `<prefix>.model.json` (parameters).

## `accmix evaluate`

Evaluate fitted models against CLIP/PIP-seq using a JSON config.

```bash
accmix evaluate -c <config.json>
```

Config keys

- `input_data_parquet` (required): Annotated input parquet used for modeling.
- `clipseq_bed` (required): CLIP-seq peaks BED.
- `pipseq_parquet` (required): PIP-seq parquet.
- `rbp_name`: RBP label.
- `motif_id`: Motif identifier.
- `motif_logo`: Path to motif logo image (optional, for plots).
- `output_root`: Output directory (default `results`).
- `model_json` (required): Path to fitted model JSON.
- `score_phastcons100_threshold`: Conservation cutoff (float, default 1.0).
- `motif_range`: Window size around motifs for evaluation (int, default 50).

Outputs: plots under `results/plots/`, logs and tables under `results/logs/`.
