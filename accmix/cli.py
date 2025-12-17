import os
import sys
import datetime as _dt
from pathlib import Path
from typing import Optional

import polars as pl
import typer

from accmix.scanner import scan_pwm
from accmix.annotate import compute_sl, annotate_tss
from accmix.model_core import fit_model
from accmix import evaluate


app = typer.Typer(help="Accessibility mixture model: scan PWM, annotate sites, and fit/evaluate EM models.")


def _timestamp() -> str:
    """Return a compact timestamp for result file names."""
    return _dt.datetime.now().strftime("%Y%m%d_%H%M%S")


@app.command("scan", help="Scan genome with PWM and compute inner/outer scores.")
def scan_pwm_cmd(
    fasta: Path = typer.Option(..., "-f", "--fasta", help="Genome FASTA (.fa or .fa.gz)."),
    pwm: Path = typer.Option(..., "-p", "--pwm", help="PWM txt with columns Pos,A,C,G,T/U (probabilities)."),
    out_prefix: Optional[Path] = typer.Option(
        None,
        "-o",
        "--out-prefix",
        help="Custom output prefix (default: results/<pwm>.<timestamp>).",
    ),
    M: int = typer.Option(50, "-M", help="Inner half-width in bp (default: 50)."),
    N: int = typer.Option(500, "-N", help="Outer half-width in bp (default: 500; must be >M)."),
    A: int = typer.Option(2_000_000, "-A", help="Top-A entries to keep per strand (default: 2e6)."),
    B: int = typer.Option(2_000_000, "-B", help="Bottom-B entries to keep per strand (default: 2e6)."),
    bg: Optional[str] = typer.Option(
        None,
        "-g",
        "--bg",
        help="Background probs for LLR as 'A,C,G,T' (overridden by --bg-from-fasta).",
    ),
    bg_from_fasta: bool = typer.Option(
        False,
        "-G",
        "--bg-from-fasta",
        help="Estimate background composition from the provided FASTA.",
    ),
) -> None:
    """Scan the genome with a PWM and write strand-aware score tables.

    This reproduces the behavior of the original Snakemake/script interface but
    exposes it as a typed CLI command.
    """

    os.makedirs("results", exist_ok=True)
    pwm_name = pwm.stem
    ts = _timestamp()
    out_prefix_str = str(out_prefix or Path("results") / f"{pwm_name}.{ts}")

    scan_pwm(
        fasta=str(fasta),
        pwm=str(pwm),
        out_prefix=out_prefix_str,
        M=M,
        N=N,
        A=A,
        B=B,
        bg=bg,
        bg_from_fasta=bg_from_fasta,
    )


@app.command("annotate-acc", help="Compute accessibility-derived s_l from AS native/fixed and PWM top table.")
def annotate_acc_cmd(
    ASnative: Path = typer.Option(..., "-n", "--ASnative", help="AS native file contains chr, position, strand, score, depth, and motif information (motif information is not used)."),
    ASfixed: Path = typer.Option(..., "-f", "--ASfixed", help="AS fixed bed6 file contains chr, start, end, strand, score, and depth."),
    toptable: Path = typer.Option(..., "-t", "--toptable", help="PWM scan top-table TSV.gz with inner/outer scores."),
    out: Optional[Path] = typer.Option(
        None,
        "-o",
        "--out",
        help="Output parquet (default: results/sl.<timestamp>.parquet).",
    ),
    M: int = typer.Option(50, "-M", help="Inner half-width used in scanning (default: 50)."),
    N: int = typer.Option(500, "-N", help="Outer half-width used in scanning (default: 500)."),
) -> None:
    """Derive accessibility score s_l around motif sites using AS native/fixed tracks."""

    os.makedirs("results", exist_ok=True)
    out_path = out or Path("results") / f"sl.{_timestamp()}.parquet"
    compute_sl(str(ASnative), str(ASfixed), str(toptable), str(out_path), M=M, N=N)
    typer.echo(f"Wrote {out_path}")


@app.command("annotate-tss", help="Add TSS proximity, conservation, and TPM annotations to sites.")
def annotate_tss_cmd(
    input_parquet: Path = typer.Option(
        ...,
        "-i",
        "--input-parquet",
        help="Input parquet with id and prior features (e.g. output of annotate-acc).",
    ),
    out: Optional[Path] = typer.Option(
        None,
        "-o",
        "--out",
        help="Output parquet (default: results/annotated.<timestamp>.parquet).",
    ),
    rna_seq_parquet: Path = typer.Option(
        Path("data/expression/ENCFF364YCB_HeLa_RNAseq_Transcripts_count_curated.parquet"),
        "-r",
        "--rna-seq-parquet",
        help="RNA-seq transcript-level expression parquet (TPM).",
    ),
    phastcons_bed: Path = typer.Option(
        Path("data/conservation_score/phastCons100way1.bed"),
        "-c",
        "--phastcons-bed",
        help="BED file with phastCons peaks used for regional conservation score.",
    ),
    phastcons_parquet: Path = typer.Option(
        Path("data/conservation_score/hg38.phastCons100way.1.parquet"),
        "-p",
        "--phastcons-parquet",
        help="Parquet with base-level phastCons scores.",
    ),
    phylop_parquet: Path = typer.Option(
        Path("data/conservation_score/hg38.phyloP100way.1.parquet"),
        "-y",
        "--phylop-parquet",
        help="Parquet with base-level phyloP scores.",
    ),
    reference: str = typer.Option(
        "GRCh38",
        "-R",
        "--reference",
        help="Reference genome assembly name (e.g. GRCh38).",
    ),
) -> None:
    """Annotate motif sites with TSS distance, conservation, and expression.

    Warning: this step can be memory-intensive on genome-scale inputs.
    """

    os.makedirs("results", exist_ok=True)
    out_path = out or Path("results") / f"annotated.{_timestamp()}.parquet"
    annotate_tss(
        input_parquet=str(input_parquet),
        out_parquet=str(out_path),
        rna_seq_parquet=str(rna_seq_parquet),
        phastcons_bed=str(phastcons_bed),
        phastcons_parquet=str(phastcons_parquet),
        phylop_parquet=str(phylop_parquet),
        reference=reference,
    )
    typer.echo(f"Wrote {out_path}")


@app.command("model", help="Fit Gaussian EM model with logistic prior on annotated sites.")
def model_cmd(
    input_parquet: Path = typer.Option(
        ...,
        "-i",
        "--input-parquet",
        help="Annotated parquet with site id, s_l column, and feature columns.",
    ),
    out_prefix: Optional[Path] = typer.Option(
        None,
        "-o",
        "--out-prefix",
        help="Output prefix (default: results/<rbp>_<motif>.<timestamp>).",
    ),
    rbp_name: str = typer.Option("RBP", "-r", "--rbp-name", help="RBP name label used in output prefix."),
    motif_id: str = typer.Option("Motif", "-m", "--motif-id", help="Motif ID label used in output prefix."),
    feat_cols: Optional[str] = typer.Option(
        None,
        "-F",
        "--feat-cols",
        help="Comma-separated list of feature column names (otherwise use default set).",
    ),
    site_col: str = typer.Option("id", "-s", "--site-col", help="Site identifier column name."),
    s_col: str = typer.Option("s_l", "-S", "--s-col", help="Accessibility score column name."),
    mode: str = typer.Option(
        "em",
        "-M",
        "--mode",
        help="Modeling mode (currently only 'em' with Gaussian components is supported).",
    ),
) -> None:
    """Fit the EM mixture model and write per-site priors/posteriors and model JSON."""

    if mode != "em":
        typer.echo("Only 'em' mode is currently supported.", err=True)
        typer.Exit(code=1)

    os.makedirs("results", exist_ok=True)
    ts = _timestamp()
    base = out_prefix or Path("results") / f"{rbp_name}_{motif_id}.{ts}"

    feat_list = None
    if feat_cols:
        feat_list = [c.strip() for c in feat_cols.split(",") if c.strip()]

    out_parquet, out_json = fit_model(
        input_parquet=str(input_parquet),
        out_prefix=str(base),
        rbp_name=rbp_name,
        motif_id=motif_id,
        feature_columns=feat_list,
        site_col=site_col,
        s_col=s_col,
    )
    typer.echo(f"Wrote {out_parquet} and {out_json}")


@app.command("evaluate", help="Evaluate fitted models against CLIP/PIP-seq data.")
def evaluate_cmd(
    model_parquet: Path = typer.Option(
        ...,
        "-M",
        "--model-parquet",
        help="Model parquet produced by 'accmix model' (contains prior/posterior).",
    ),
    clipseq_bed: Path = typer.Option(
        ...,
        "-b",
        "--clipseq-bed",
        help="CLIP-seq peaks BED path.",
    ),
    pipseq_parquet: Path = typer.Option(
        ...,
        "-p",
        "--pipseq-parquet",
        help="PIP-seq parquet path.",
    ),
    rbp_name: str = typer.Option("RBP", "-r", "--rbp-name", help="RBP label for titles/filenames."),
    motif_id: str = typer.Option("Motif", "-m", "--motif-id", help="Motif ID for titles/filenames."),
    output_root: Path = typer.Option(Path("results"), "-o", "--output-root", help="Directory to write plots/logs."),
    motif_logo: Optional[Path] = typer.Option(None, "-L", "--motif-logo", help="Optional motif logo image."),
    score_phastcons100_threshold: float = typer.Option(1.0, "-t", "--score-phastcons100-threshold", help="Conservation score threshold."),
    motif_range: int = typer.Option(50, "-R", "--motif-range", help="Half-window around site position for overlaps."),
    name_add: str = typer.Option("", "-n", "--name-add", help="Additional string to append to generated filenames."),
) -> None:
    """Run the evaluation pipeline using direct CLI options."""

    evaluate.run_evaluation(
        model_parquet=str(model_parquet),
        clipseq_bed=str(clipseq_bed),
        pipseq_parquet=str(pipseq_parquet),
        rbp_name=rbp_name,
        motif_id=motif_id,
        output_root=str(output_root),
        motif_logo=str(motif_logo) if motif_logo else None,
        score_phastcons100_threshold=score_phastcons100_threshold,
        motif_range=motif_range,
        name_add=name_add,
    )


def main() -> None:
    """Entry point for the `accmix` command."""
    app()
