import os
import sys
import datetime as _dt
import polars as pl
from accmix.scanner import scan_pwm
from accmix.annotate import annotate, compute_sl, annotate_tss
from accmix.model_core import fit_model


def _timestamp():
    return _dt.datetime.now().strftime("%Y%m%d_%H%M%S")


def scan_pwm_cmd(args=None):
    import argparse
    ap = argparse.ArgumentParser(
        prog="accmix scan",
        description=(
            "Scan genome with a PWM, computing strand-aware inner/outer region scores around each center. "
            "Outputs gzipped TSVs under results/ with informative prefixes."
        ),
    )
    ap.add_argument("--fasta", required=True, help="Genome FASTA (.fa or .fa.gz)")
    ap.add_argument("--pwm", required=True, help="PWM txt with columns Pos,A,C,G,T/U (probabilities)")
    ap.add_argument(
        "--out-prefix",
        default=None,
        help="Custom output prefix (default: results/<pwm>.<timestamp>)",
    )
    ap.add_argument("-M", type=int, default=50, help="Inner half-width (default: 50)")
    ap.add_argument("-N", type=int, default=500, help="Outer half-width (default: 500; must be >M)")
    ap.add_argument("-A", type=int, default=2_000_000, help="Top-A entries to keep per strand (default: 2e6)")
    ap.add_argument("-B", type=int, default=2_000_000, help="Bottom-B entries to keep per strand (default: 2e6)")
    ap.add_argument(
        "--bg",
        type=str,
        default=None,
        help="Background probs for LLR as 'A,C,G,T' (overridden by --bg-from-fasta)",
    )
    ap.add_argument(
        "--bg-from-fasta",
        action="store_true",
        help="Estimate background composition from the provided FASTA",
    )
    args = ap.parse_args(args=args)

    os.makedirs("results", exist_ok=True)
    pwm_name = os.path.splitext(os.path.basename(args.pwm))[0]
    ts = _timestamp()
    out_prefix = args.out_prefix or f"results/{pwm_name}.{ts}"

    sys.argv = [
        "scan_pwm_regions.py",
        "--fasta", args.fasta,
        "--pwm", args.pwm,
        "-M", str(args.M),
        "-N", str(args.N),
        "-A", str(args.A),
        "-B", str(args.B),
        "--out-prefix", out_prefix,
    ] + (["--bg", args.bg] if args.bg else []) + (["--bg-from-fasta"] if args.bg_from_fasta else [])
    # Call internal function
    scan_pwm(
        fasta=args.fasta,
        pwm=args.pwm,
        out_prefix=out_prefix,
        M=args.M,
        N=args.N,
        A=args.A,
        B=args.B,
        bg=args.bg,
        bg_from_fasta=args.bg_from_fasta,
    )


def annotate_acc_cmd(args=None):
    import argparse
    ap = argparse.ArgumentParser(
        prog="accmix annotate-acc",
        description=(
            "Compute accessibility-derived s_l from AS native/fixed bed6 and PWM top table (no TSS/conservation)."
        ),
    )
    ap.add_argument("--ASnative", required=True, help="AS native bed6 file")
    ap.add_argument("--ASfixed", required=True, help="AS fixed bed6 file")
    ap.add_argument(
        "--toptable",
        required=True,
        help="Top table TSV.gz from scan step with inner/outer PWM features",
    )
    ap.add_argument(
        "--out",
        default=None,
        help="Output parquet (default: results/sl.<timestamp>.parquet)",
    )
    ap.add_argument("--M", type=int, default=50, help="Inner half-width (default 50)")
    ap.add_argument("--N", type=int, default=500, help="Outer half-width (default 500)")
    args = ap.parse_args(args=args)

    os.makedirs("results", exist_ok=True)
    out_path = args.out or f"results/sl.{_timestamp()}.parquet"
    compute_sl(args.ASnative, args.ASfixed, args.toptable, out_path, M=args.M, N=args.N)
    print(f"Wrote {out_path}")


def annotate_tss_cmd(args=None):
    import argparse
    ap = argparse.ArgumentParser(
        prog="accmix annotate-tss",
        description=(
            "Annotate sites with transcript/TSS proximity, conservation (phastCons/phyloP), and TPM."
        ),
    )
    ap.add_argument(
        "--input-parquet",
        required=True,
        help="Parquet containing id and prior features (e.g., output of annotate-acc)",
    )
    ap.add_argument(
        "--out",
        default=None,
        help="Output parquet (default: results/annotated.<timestamp>.parquet)",
    )
    ap.add_argument("--rna-seq-parquet", default="../metadatasets/ENCFF364YCB_HeLa_RNAseq_Transcripts_count_curated.parquet")
    ap.add_argument("--phastcons-bed", default="../metadatasets/phastCons100way1.bed")
    ap.add_argument("--phastcons-parquet", default="../midway3_results/hg38.phastCons100way.1.parquet")
    ap.add_argument("--phylop-parquet", default="../midway3_results/hg38.phyloP100way.1.parquet")
    ap.add_argument("--reference", default="GRCh38")
    args = ap.parse_args(args=args)

    os.makedirs("results", exist_ok=True)
    out_path = args.out or f"results/annotated.{_timestamp()}.parquet"
    annotate_tss(
        input_parquet=args.input_parquet,
        out_parquet=out_path,
        rna_seq_parquet=args.rna_seq_parquet,
        phastcons_bed=args.phastcons_bed,
        phastcons_parquet=args.phastcons_parquet,
        phylop_parquet=args.phylop_parquet,
        reference=args.reference,
    )
    print(f"Wrote {out_path}")


def model_cmd(args=None):
    import argparse
    ap = argparse.ArgumentParser(
        prog="accmix model",
        description=(
            "Run Gaussian EM mixture with logistic prior over features on an annotated parquet. "
            "Writes model parameters (JSON) and per-site priors/posteriors (parquet)."
        ),
    )
    ap.add_argument("--input-parquet", required=True, help="Annotated parquet with site id, s column, and features")
    ap.add_argument(
        "--out-prefix",
        default=None,
        help="Output prefix (default: results/<rbp>_<motif>.<timestamp>)",
    )
    ap.add_argument("--rbp-name", default="RBP", help="Label for RBP name (used in output prefix)")
    ap.add_argument("--motif-id", default="Motif", help="Label for motif ID (used in output prefix)")
    # Script-equivalent knobs
    ap.add_argument("--feat-cols", default=None,
                    help="Comma-separated feature columns to use (default set if omitted)")
    ap.add_argument("--site-col", default="id", help="Site id column name (default: id)")
    ap.add_argument("--s-col", default="s_l", help="s_l column name (default: s_l)")
    ap.add_argument("--mode", choices=["em"], default="em",
                    help="Modeling mode (currently supports em with Gaussian components)")
    args = ap.parse_args(args=args)

    os.makedirs("results", exist_ok=True)
    ts = _timestamp()
    base = args.out_prefix or f"results/{args.rbp_name}_{args.motif_id}.{ts}"

    feat_cols = None
    if args.feat_cols:
        feat_cols = [c.strip() for c in args.feat_cols.split(",") if c.strip()]

    out_parquet, out_json = fit_model(
        input_parquet=args.input_parquet,
        out_prefix=base,
        rbp_name=args.rbp_name,
        motif_id=args.motif_id,
        feature_columns=feat_cols,
        site_col=args.site_col,
        s_col=args.s_col,
    )
    print(f"Wrote {out_parquet} and {out_json}")


def main():
    import argparse
    parser = argparse.ArgumentParser(
        prog="accmix",
        description=(
            "Accessibility mixture model: scan PWM, annotate sites (optionally compute s_l), and fit EM model."
        ),
    )
    subparsers = parser.add_subparsers(dest="cmd", required=True)

    subparsers.add_parser(
        "scan",
        help="Scan genome with PWM and output strand-aware inner/outer region scores",
    )
    subparsers.add_parser(
        "annotate-acc",
        help="Compute accessibility-derived s_l from AS native/fixed and PWM top table",
    )
    subparsers.add_parser(
        "annotate-tss",
        help="Add transcript/TSS proximity, conservation, and TPM to sites",
    )
    subparsers.add_parser(
        "model",
        help="Fit Gaussian EM with logistic prior; outputs model JSON and site parquet",
    )

    # Parse only the top-level to dispatch; pass remaining to subcommands
    if len(sys.argv) <= 1:
        parser.print_help()
        sys.exit(1)
    cmd = sys.argv[1]
    args = sys.argv[2:]
    if cmd == "scan":
        scan_pwm_cmd(args)
    elif cmd == "annotate-acc":
        annotate_acc_cmd(args)
    elif cmd == "annotate-tss":
        annotate_tss_cmd(args)
    elif cmd == "model":
        model_cmd(args)
    else:
        parser.print_help()
        sys.exit(2)
