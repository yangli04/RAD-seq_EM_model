"""Memory-optimized annotate variant with BigWig support, keeping the original annotate.py intact.

This module provides:
- compute_sl(): supports BED6 mode, BigWig pair mode (plus,minus for native/fixed),
  and single BigWig per condition (strand-agnostic)
- annotate_tss(): memory-bounded per-chromosome processing with optional BigWig fast path
"""

from __future__ import annotations
import argparse
from typing import Optional
import os
import polars as pl
import pyranges as pr
import math
from pathlib import Path


def compute_sl(
    ASnative: str,
    ASfixed: str,
    toptable: str,
    out_parquet: str,
    M: int = 50,
    N: int = 500,
    threshold: int = 8,
) -> None:
    """Compute s_l using either BED6 inputs or BigWig files for native/fixed.

    BigWig options:
    - Pair mode (strand-specific):
        ASnative="/path/native_plus.bw,/path/native_minus.bw"
        ASfixed="/path/fixed_plus.bw,/path/fixed_minus.bw"
      Uses the row strand to select plus vs minus BigWig.
    - Single mode (strand-agnostic):
        ASnative="/path/native.bw"
        ASfixed="/path/fixed.bw"
      Ignores strand and always samples from the given BigWig.

    In BigWig modes, s_l is computed via per-site inner/outer means and variances.
        Memory is bounded by iterating per chromosome.
    """

    dftop = (
        pl.read_csv(toptable, separator="\t")
        .with_columns((pl.col("inner_mean_logPWM") - pl.col("outer_mean_logPWM")).alias("diff_logPWM"))
        .sort(["chrom", "pos1", "strand"])  # ensure per-chrom streaming order
        .with_columns(pl.concat_str(["chrom", "pos1", "strand"], separator="_").alias("id"))
    )

    # Detect BigWig mode(s)
    def _is_bw(s: str) -> bool:
        s = s or ""
        return s.endswith(".bw") or s.endswith(".bigwig") or "," in s

    if _is_bw(ASnative) and _is_bw(ASfixed):
        # Normalize inputs into lists
        nat_parts = [p for p in ASnative.split(",") if p]
        fix_parts = [p for p in ASfixed.split(",") if p]

        try:
            import pyBigWig  # type: ignore
            import numpy as np
        except Exception as e:
            raise SystemExit(f"pyBigWig is required for BigWig mode: {e}")

        results_rows = []

        def _vals(bw, chrom, st, en):
            try:
                arr = bw.values(chrom, int(st), int(en), numpy=True)
                if arr is None:
                    return np.array([], dtype=float)
                return arr[np.isfinite(arr)].astype(float)
            except Exception:
                return np.array([], dtype=float)

        def _t_stat(inner: "np.ndarray", outer: "np.ndarray") -> float:
            n1 = inner.size
            n2 = outer.size
            if n1 < threshold or n2 < threshold:
                return float("nan")
            m1 = float(inner.mean())
            m2 = float(outer.mean())
            v1 = float(inner.var(ddof=0))
            v2 = float(outer.var(ddof=0))
            denom = v1 / max(n1, 1) + v2 / max(n2, 1)
            return (m1 - m2) / denom if denom > 0 else float("nan")

        if len(nat_parts) == 2 and len(fix_parts) == 2:
            # Strand-specific pair mode
            nat_plus, nat_minus = nat_parts
            fix_plus, fix_minus = fix_parts
            bw_nat_plus = pyBigWig.open(nat_plus)
            bw_nat_minus = pyBigWig.open(nat_minus)
            bw_fix_plus = pyBigWig.open(fix_plus)
            bw_fix_minus = pyBigWig.open(fix_minus)

            for chrom in dftop["chrom"].unique().to_list():
                sites = dftop.filter(pl.col("chrom") == chrom)
                if sites.height == 0:
                    continue
                for row in sites.iter_rows(named=True):
                    pos = int(row["pos1"])
                    strand = row["strand"]
                    start_inner = max(0, pos - M); end_inner = pos + M
                    start_left = max(0, pos - N); end_left = max(0, pos - M)
                    start_right = pos + M; end_right = pos + N
                    nat_bw = bw_nat_plus if strand == "+" else bw_nat_minus
                    fix_bw = bw_fix_plus if strand == "+" else bw_fix_minus
                    nat_inner = _vals(nat_bw, chrom, start_inner, end_inner)
                    nat_outer = np.concatenate([
                        _vals(nat_bw, chrom, start_left, end_left),
                        _vals(nat_bw, chrom, start_right, end_right),
                    ])
                    fix_inner = _vals(fix_bw, chrom, start_inner, end_inner)
                    fix_outer = np.concatenate([
                        _vals(fix_bw, chrom, start_left, end_left),
                        _vals(fix_bw, chrom, start_right, end_right),
                    ])
                    t_native = _t_stat(nat_inner, nat_outer)
                    t_fixed = _t_stat(fix_inner, fix_outer)
                    s_l = ((0.0 if math.isnan(t_native) else t_native ** 2)
                           + (0.0 if math.isnan(t_fixed) else t_fixed ** 2))
                    results_rows.append({"id": row["id"], "t_native": t_native, "t_fixed": t_fixed, "s_l": s_l})

            try:
                bw_nat_plus.close(); bw_nat_minus.close(); bw_fix_plus.close(); bw_fix_minus.close()
            except Exception:
                pass

        elif len(nat_parts) == 1 and len(fix_parts) == 1:
            # Single BigWig per condition (strand-agnostic)
            bw_nat = pyBigWig.open(nat_parts[0])
            bw_fix = pyBigWig.open(fix_parts[0])

            for chrom in dftop["chrom"].unique().to_list():
                sites = dftop.filter(pl.col("chrom") == chrom)
                if sites.height == 0:
                    continue
                for row in sites.iter_rows(named=True):
                    pos = int(row["pos1"])
                    start_inner = max(0, pos - M); end_inner = pos + M
                    start_left = max(0, pos - N); end_left = max(0, pos - M)
                    start_right = pos + M; end_right = pos + N
                    nat_inner = _vals(bw_nat, chrom, start_inner, end_inner)
                    nat_outer = np.concatenate([
                        _vals(bw_nat, chrom, start_left, end_left),
                        _vals(bw_nat, chrom, start_right, end_right),
                    ])
                    fix_inner = _vals(bw_fix, chrom, start_inner, end_inner)
                    fix_outer = np.concatenate([
                        _vals(bw_fix, chrom, start_left, end_left),
                        _vals(bw_fix, chrom, start_right, end_right),
                    ])
                    t_native = _t_stat(nat_inner, nat_outer)
                    t_fixed = _t_stat(fix_inner, fix_outer)
                    s_l = ((0.0 if math.isnan(t_native) else t_native ** 2)
                           + (0.0 if math.isnan(t_fixed) else t_fixed ** 2))
                    results_rows.append({"id": row["id"], "t_native": t_native, "t_fixed": t_fixed, "s_l": s_l})

            try:
                bw_nat.close(); bw_fix.close()
            except Exception:
                pass

        else:
            raise SystemExit("BigWig mode must be either pair (plus,minus for both) or single (one file per condition).")

        results_df = pl.DataFrame(results_rows)
        pwm_columns = [
            "inner_mean_logPWM", "outer_mean_logPWM",
            "GC_inner_pct", "GC_outer_pct",
            "kmer_count_inner", "kmer_count_outer",
            "diff_logPWM",
        ]
        results_df = results_df.join(dftop.select(["id"] + pwm_columns), on="id", how="left")

    else:
        # BED6 mode (memory-bounded): process per chromosome using lazy CSV scans
        dftop_inner = (
            dftop.with_columns((pl.col("pos1") - pl.lit(M)).alias("Start"), (pl.col("pos1") + pl.lit(M)).alias("End"))
            .rename({"chrom": "Chromosome", "strand": "Strand"})
        )
        dftop_outer = (
            dftop.with_columns((pl.col("pos1") - pl.lit(N)).alias("Start"), (pl.col("pos1") + pl.lit(N)).alias("End"))
            .rename({"chrom": "Chromosome", "strand": "Strand"})
        )

        # Lazy scan of AS files to slice by chromosome
        ASfixed_scan = pl.scan_csv(
            ASfixed,
            has_header=False,
            separator="\t",
            new_columns=["Chromosome", "Start", "Strand", "AS_fixed", "depth_fixed", "motif"],
        ).select(["Chromosome", "Start", "Strand", "AS_fixed", "depth_fixed"])  # drop unused

        ASnative_scan = pl.scan_csv(
            ASnative,
            has_header=False,
            separator="\t",
            new_columns=["Chromosome", "Start", "Strand", "AS_native", "depth_native", "motif"],
        ).select(["Chromosome", "Start", "Strand", "AS_native", "depth_native"])  # drop unused

        results_parts: list[pl.DataFrame] = []

        def summarize_weighted_by_id(
            df: pl.DataFrame,
            value_col: str,
            weight_col: str,
            *,
            id_col: str = "id",
            threshold: int = threshold,
            ddof: int = 0,
        ) -> pl.DataFrame:
            val = pl.col(value_col)
            w = pl.col(weight_col)
            w_mean_name = f"{value_col}_w_mean"
            res = (
                df
                .group_by(id_col)
                .agg([
                    pl.len().alias("n"),
                    val.mean().alias(f"{value_col}_mean"),
                    val.var(ddof=ddof).alias(f"{value_col}_var"),
                    (val * w).sum().alias("_wx"),
                    w.sum().alias("_w"),
                    (w * val.pow(2)).sum().alias("_wx2"),
                    pl.first("inner_mean_logPWM"),
                    pl.first("outer_mean_logPWM"),
                    pl.first("GC_inner_pct"),
                    pl.first("GC_outer_pct"),
                    pl.first("kmer_count_inner"),
                    pl.first("kmer_count_outer"),
                    pl.first("diff_logPWM"),
                ])
                .filter(pl.col("n") >= threshold)
                .with_columns([
                    (pl.col("_wx") / pl.col("_w")).alias(w_mean_name),
                ])
                .with_columns(
                    (pl.col("_wx2") / pl.col("_w") - pl.col(w_mean_name).pow(2)).alias(f"{value_col}_w_var"),
                )
                .drop(["_wx", "_wx2"])
            )
            return res

        def filter_outer(df: pl.DataFrame) -> pl.DataFrame:
            flank = max(N - M, 0)
            mask1 = (df["Start_b"] >= df["Start"]) & (df["Start_b"] <= df["Start"] + flank)
            mask2 = (df["Start_b"] >= df["End"] - flank) & (df["Start_b"] <= df["End"])  
            return df.filter(mask1 | mask2)

        for chrom in dftop["chrom"].unique().to_list():
            # Slice toptable windows for this chromosome
            inner_ch = dftop_inner.filter(pl.col("Chromosome") == chrom)
            outer_ch = dftop_outer.filter(pl.col("Chromosome") == chrom)
            if inner_ch.height == 0 and outer_ch.height == 0:
                continue

            # Collect AS for this chromosome lazily
            AS_fixed_ch = (
                ASfixed_scan.filter(pl.col("Chromosome") == chrom)
                .with_columns((pl.col("Start") + 1).alias("End"))
                .collect(streaming=True)
            )
            AS_native_ch = (
                ASnative_scan.filter(pl.col("Chromosome") == chrom)
                .with_columns((pl.col("Start") + 1).alias("End"))
                .collect(streaming=True)
            )
            if AS_fixed_ch.height == 0 or AS_native_ch.height == 0:
                continue

            # Build PyRanges for overlaps (chrom-local)
            inner_pr = pr.PyRanges(inner_ch.to_pandas())
            outer_pr = pr.PyRanges(outer_ch.to_pandas())
            fixed_pr = pr.PyRanges(AS_fixed_ch.to_pandas())
            native_pr = pr.PyRanges(AS_native_ch.to_pandas())

            inner_fixed_pr = inner_pr.join_overlaps(fixed_pr, strand_behavior="same", join_type="inner")
            outer_fixed_pr = outer_pr.join_overlaps(fixed_pr, strand_behavior="same", join_type="inner")
            inner_native_pr = inner_pr.join_overlaps(native_pr, strand_behavior="same", join_type="inner")
            outer_native_pr = outer_pr.join_overlaps(native_pr, strand_behavior="same", join_type="inner")

            inner_fixed_df = pl.from_pandas(inner_fixed_pr)
            inner_native_df = pl.from_pandas(inner_native_pr)
            outer_fixed_df = filter_outer(pl.from_pandas(outer_fixed_pr))
            outer_native_df = filter_outer(pl.from_pandas(outer_native_pr))

            # Summaries per id
            outer_fixed_sum = summarize_weighted_by_id(outer_fixed_df, value_col="AS_fixed", weight_col="depth_fixed")
            outer_native_sum = summarize_weighted_by_id(outer_native_df, value_col="AS_native", weight_col="depth_native")
            inner_fixed_sum = summarize_weighted_by_id(inner_fixed_df, value_col="AS_fixed", weight_col="depth_fixed")
            inner_native_sum = summarize_weighted_by_id(inner_native_df, value_col="AS_native", weight_col="depth_native")

            pwm_columns = ['inner_mean_logPWM', 'outer_mean_logPWM', 'GC_inner_pct', 'GC_outer_pct', 'kmer_count_inner', 'kmer_count_outer', 'diff_logPWM']

            fixed_part = (
                outer_fixed_sum.join(inner_fixed_sum, on="id", how="inner", suffix="_inner")
                .drop([i+"_inner" for i in pwm_columns])
                .with_columns(
                    (
                        (pl.col("AS_fixed_mean_inner") - pl.col("AS_fixed_mean")) /
                        (pl.col("AS_fixed_var_inner")/pl.col("_w_inner") + pl.col("AS_fixed_var")/pl.col("_w"))
                    ).alias("t_fixed")
                )
            )
            native_part = (
                outer_native_sum.join(inner_native_sum, on="id", how="inner", suffix="_inner")
                .drop([i+"_inner" for i in pwm_columns])
                .with_columns(
                    (
                        (pl.col("AS_native_mean_inner") - pl.col("AS_native_mean")) /
                        (pl.col("AS_native_var_inner")/pl.col("_w_inner") + pl.col("AS_native_var")/pl.col("_w"))
                    ).alias("t_native")
                )
            )

            results_ch = (
                fixed_part.join(native_part, on="id", suffix="_native")
                .with_columns((pl.col("t_native").pow(2) + pl.col("t_fixed").pow(2)).alias("s_l"))
            )
            results_parts.append(results_ch)

        # Combine chromosome parts
        results_df = pl.concat(results_parts) if results_parts else pl.DataFrame({"id": [], "s_l": []})

    # Sort sites by genomic order (chrom, position, strand) before writing
    results_df = (
        results_df
        .with_columns([
            pl.col("id").str.split("_").alias("_id_split")
        ])
        .with_columns([
            pl.col("_id_split").list.get(0).alias("_chrom"),
            pl.col("_id_split").list.get(1).cast(pl.Int64).alias("_pos"),
            pl.col("_id_split").list.get(2).alias("_strand"),
        ])
        .sort(["_chrom", "_pos", "_strand"])  # preserves within-chrom numeric order
        .drop(["_id_split", "_chrom", "_pos", "_strand"])  # clean up temps
    )
    results_df.write_parquet(out_parquet)


def annotate_tss(
    input_parquet: str,
    out_parquet: str,
    rna_seq_parquet: str = "data/expression/ENCFF364YCB_HeLa_RNAseq_Transcripts_count_curated.parquet",
    phastcons_bed: str = "data/conservation_score/phastCons100way1.bed",
    phastcons_parquet: str = "data/conservation_score/hg38.phastCons100way.1.parquet",
    phylop_parquet: str = "data/conservation_score/hg38.phyloP100way.1.parquet",
    reference: str = "GRCh38",
    *,
    # Optional BigWig fast path (if provided, skips Parquet scanning for scores)
    phastcons_bigwig: Optional[str] = None,
    phylop_bigwig: Optional[str] = None,
    # Streaming tile size and buffer for Parquet path
    tile_mb: int = 5,
    window_buffer: int = 100_000,
):
    """Memory-optimized TSS annotation that scales to all chromosomes.

    - If BigWig files are provided, fetch per-site conservation scores (fastest, lowest memory).
    - Otherwise, stream by chromosome in tiles (tile_mb) and scan conservation Parquet only for
      the current window ± buffer, writing chunk outputs and combining with a final deterministic sort.
    """
    from metagene import load_reference, map_to_transcripts
    # Helpers
    def _drop_if_exists(df: pl.DataFrame | pl.LazyFrame, cols: list[str]):
        if isinstance(df, pl.LazyFrame):
            # For LazyFrame, drop ignores missing columns gracefully
            return df.drop([c for c in cols])
        existing = [c for c in cols if c in df.columns]
        return df.drop(existing) if existing else df

    # Load static resources once
    RNA_seq_data = pl.read_parquet(rna_seq_parquet)
    conservative_df = pl.read_csv(
        phastcons_bed, separator="\t", has_header=False,
        new_columns=["Chromosome", "Start", "End", "Name"],
        comment_prefix="#",
    ).with_columns(pl.col("Chromosome").str.replace('chr', ''))
    conservative_pr = pr.PyRanges(conservative_df.to_pandas())
    reference_df = load_reference(reference)

    # Load all sites and parse IDs (IDENTICAL to original)
    results_df = pl.read_parquet(input_parquet)
    ann_df = results_df.with_columns([
        pl.col("id").str.split("_").alias("id_split")
    ]).with_columns([
        pl.col("id_split").list.get(0).str.replace('chr', '').alias("Chromosome"),
        pl.col("id_split").list.get(2).alias("Strand"),
        pl.col("id_split").list.get(1).cast(pl.Int64).alias("Start")
    ]).drop("id_split").with_columns((pl.col("Start") + 50).alias("End"))\
      .with_columns((pl.col("Start") - 50).alias("Start"))

    # Map to transcripts (IDENTICAL to original)
    ann_df_tx = map_to_transcripts(ann_df, reference_df)
    ann_df_tx = ann_df_tx.with_columns([
        (1 / (1 + pl.col("transcript_start")/1000)).fill_null(0).alias("TSS_proximity"),
        (1 / (1 + pl.col("transcript_start") / pl.col("transcript_length")).fill_null(0)).alias("Tx_percent")
    ])
    
    # PyRanges overlap (IDENTICAL to original)
    ann_df_tx_pr = pr.PyRanges(ann_df_tx.to_pandas())
    joined_pr = ann_df_tx_pr.join_overlaps(conservative_pr, strand_behavior="ignore", join_type="left")
    joined_ann_df = pl.DataFrame(joined_pr)
    
    # Compute PhastCons100_percent and join RNA-seq (IDENTICAL to original)
    joined_ann_df = (
        joined_ann_df
        .with_columns(
            (
                ((pl.min_horizontal("End", "End_b") - pl.max_horizontal("Start", "Start_b")) / 100)
                .clip(lower_bound=0)
                .alias("PhastCons100_percent")
            )
        )
        .join(RNA_seq_data, on="transcript_id", how="left")
        .with_columns(pl.col("TPM").fill_null(0))
    )

    # Prepare chrom and start for conservation join
    joined_ann_df = joined_ann_df.with_columns(
        pl.concat_str(pl.lit("chr"), pl.col("Chromosome")).alias("chrom"), 
        (pl.col("Start") + 50).alias("start")
    )

    # OPTIMIZATION: Process per chromosome to bound memory
    use_bigwig = phastcons_bigwig is not None and phylop_bigwig is not None
    chrom_results = []
    
    for chrom in joined_ann_df["chrom"].unique().sort():
        chrom_sites = joined_ann_df.filter(pl.col("chrom") == chrom)
        if chrom_sites.height == 0:
            continue
        
        if use_bigwig:
            # BigWig: vectorized interval fetching per chromosome
            try:
                import pyBigWig  # type: ignore
                import numpy as np
                bw_phast = pyBigWig.open(phastcons_bigwig)
                bw_phyl = pyBigWig.open(phylop_bigwig)

                positions = chrom_sites["start"].to_numpy()
                vals_pc = []
                vals_pl = []
                
                for pos in positions:
                    try:
                        pc = bw_phast.values(chrom, int(pos), int(pos)+1, numpy=True)
                        pl_v = bw_phyl.values(chrom, int(pos), int(pos)+1, numpy=True)
                        vals_pc.append(float(pc[0]) if pc is not None and len(pc) > 0 and not math.isnan(pc[0]) else None)
                        vals_pl.append(float(pl_v[0]) if pl_v is not None and len(pl_v) > 0 and not math.isnan(pl_v[0]) else None)
                    except Exception:
                        vals_pc.append(None)
                        vals_pl.append(None)
                
                chrom_result = chrom_sites.with_columns([
                    pl.Series("score_phastcons100", vals_pc),
                    pl.Series("score_phylop100", vals_pl),
                ])
                chrom_results.append(chrom_result)
            except Exception as e:
                raise RuntimeError(f"BigWig error on {chrom}: {e}")
        else:
            # Parquet: scan only this chromosome's conservation data
            min_pos = max(0, chrom_sites["start"].min() - window_buffer)
            max_pos = chrom_sites["start"].max() + window_buffer
            
            phast_chrom = (
                pl.scan_parquet(phastcons_parquet)
                .filter(
                    (pl.col("chrom") == chrom) & 
                    (pl.col("start") >= min_pos) & 
                    (pl.col("start") <= max_pos)
                )
            )
            phyl_chrom = (
                pl.scan_parquet(phylop_parquet)
                .filter(
                    (pl.col("chrom") == chrom) & 
                    (pl.col("start") >= min_pos) & 
                    (pl.col("start") <= max_pos)
                )
            )
            
            # Join for this chromosome only
            chrom_result = (
                chrom_sites.lazy()
                .sort("start")
                .join_asof(phast_chrom, on="start", strategy="backward", suffix="_phastcons")
                .filter(pl.col("start") <= pl.col("end"))
                .join_asof(phyl_chrom, on="start", strategy="backward", suffix="_phylop100")
                .filter(pl.col("start") <= pl.col("end_phylop100"))
                .rename({"score": "score_phastcons100"})
                .collect(streaming=True)
            )
            chrom_results.append(chrom_result)
    
    # Combine all chromosomes
    joined_ann_joined_df = pl.concat(chrom_results).lazy()
    
    # Clean up columns (same as original)
    joined_ann_joined_df = (
        joined_ann_joined_df if isinstance(joined_ann_joined_df, pl.LazyFrame) else joined_ann_joined_df.lazy()
    ).drop([
        "Chromosome", "Start", "End", "Strand", "gene_id", "transcript_id", 
        "transcript_start", "transcript_end", "transcript_length", 
        'start_codon_pos', 'stop_codon_pos', 'exon_number', 'record_id', 
        'Start_exon', 'End_exon', 'Start_b', 'End_b', 'Name', 
        'end', 'end_phylop100', 'start', 'chrom',
        'chrom_phastcons', 'chrom_phylop100'
    ])

    # Final genomic sort
    joined_ann_joined_df = (
        joined_ann_joined_df
        .with_columns([
            pl.col("id").str.split("_").alias("_id_split")
        ])
        .with_columns([
            pl.col("_id_split").list.get(0).alias("_chrom"),
            pl.col("_id_split").list.get(1).cast(pl.Int64).alias("_pos"),
            pl.col("_id_split").list.get(2).alias("_strand"),
        ])
        .sort(["_chrom", "_pos", "_strand"])
        .drop(["_id_split", "_chrom", "_pos", "_strand"])
    )
    joined_ann_joined_df.sink_parquet(out_parquet)


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="accmix annotate_new")
    # annotate-acc parameters
    ap.add_argument("--ASnative", default=None, help="Accessibility native bed6 path OR 'plus.bw,minus.bw'")
    ap.add_argument("--ASfixed", default=None, help="Accessibility fixed bed6 path OR 'plus.bw,minus.bw'")
    ap.add_argument("--toptable", default=None, help="TopA PWM prior tsv.gz from accmix scan")
    ap.add_argument("--M", type=int, default=50, help="Inner half-width (default 50)")
    ap.add_argument("--N", type=int, default=500, help="Outer half-width (default 500)")
    ap.add_argument("--threshold", type=int, default=8, help="Minimum samples per window (default 8)")
    # annotate-tss parameters
    ap.add_argument("--input-parquet", default=None, help="Parquet to annotate (e.g., output of annotate-acc)")
    ap.add_argument("--out", default=None, help="Output parquet path for annotated features")
    ap.add_argument("--rna-seq-parquet", default="data/expression/ENCFF364YCB_HeLa_RNAseq_Transcripts_count_curated.parquet")
    ap.add_argument("--phastcons-bed", default="data/conservation_score/phastCons100way1.bed")
    ap.add_argument("--phastcons-parquet", default="data/conservation_score/hg38.phastCons100way.1.parquet")
    ap.add_argument("--phylop-parquet", default="data/conservation_score/hg38.phyloP100way.1.parquet")
    ap.add_argument("--phastcons-bigwig", default=None, help="Optional BigWig for phastCons values (fast, low memory)")
    ap.add_argument("--phylop-bigwig", default=None, help="Optional BigWig for phyloP values (fast, low memory)")
    ap.add_argument("--tile-mb", type=int, default=5, help="Tile size in megabases for streaming Parquet scan")
    ap.add_argument("--window-buffer", type=int, default=100000, help="Extra bases to include around each tile")
    ap.add_argument("--reference", default="GRCh38")
    return ap
