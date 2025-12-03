from __future__ import annotations
import argparse
from typing import Optional
import os
import polars as pl
import pyranges as pr

def compute_sl(ASnative: str, ASfixed: str, toptable: str, out_parquet: str,
               M: int = 50, N: int = 500, threshold: int = 8) -> None:
    AS_fixed = pl.read_csv(ASfixed, has_header=False,
                           new_columns=["Chromosome", "Start", "Strand", "AS_fixed", 'depth_fixed', "motif"],
                           separator="\t")\
                   .select(["Chromosome", "Start", "Strand", "AS_fixed", 'depth_fixed'])\
                   .with_columns((pl.col("Start") + 1).alias("End"))
    AS_native = pl.read_csv(ASnative, has_header=False,
                            new_columns=["Chromosome", "Start", "Strand", "AS_native", 'depth_native', "motif"],
                            separator="\t")\
                    .select(["Chromosome", "Start", "Strand", "AS_native", 'depth_native'])\
                    .with_columns((pl.col("Start") + 1).alias("End"))
    AS_fixed_pr = pr.PyRanges(AS_fixed.to_pandas())
    AS_native_pr = pr.PyRanges(AS_native.to_pandas())
    dftop = pl.read_csv(toptable, separator="\t")\
                .with_columns((pl.col("inner_mean_logPWM") - pl.col("outer_mean_logPWM")).alias("diff_logPWM"))\
                .sort(['chrom', 'pos1', 'strand'])\
                .with_columns(pl.concat_str(['chrom', 'pos1', 'strand'], separator="_").alias("id"))

    dftop_inner = dftop.with_columns((pl.col("pos1") - pl.lit(M)).alias("Start"), (pl.col("pos1") + pl.lit(M)).alias("End"))\
                       .rename({"chrom": "Chromosome", "strand": "Strand"})
    dftop_outer = dftop.with_columns((pl.col("pos1") - pl.lit(N)).alias("Start"), (pl.col("pos1") + pl.lit(N)).alias("End"))\
                       .rename({"chrom": "Chromosome", "strand": "Strand"})
    dftop_inner_pr = pr.PyRanges(dftop_inner.to_pandas())
    dftop_outer_pr = pr.PyRanges(dftop_outer.to_pandas())

    # dftop_inner_fixed_pr = dftop_inner_pr.join(AS_fixed_pr, strandedness="same")
    # dftop_outer_fixed_pr = dftop_outer_pr.join(AS_fixed_pr, strandedness="same")
    # dftop_inner_native_pr = dftop_inner_pr.join(AS_native_pr, strandedness="same")
    # dftop_outer_native_pr = dftop_outer_pr.join(AS_native_pr, strandedness="same")
    dftop_inner_fixed_pr = dftop_inner_pr.join_overlaps(AS_fixed_pr, strand_behavior="same", join_type = "inner")
    dftop_outer_fixed_pr = dftop_outer_pr.join_overlaps(AS_fixed_pr, strand_behavior="same", join_type = "inner")
    dftop_inner_native_pr = dftop_inner_pr.join_overlaps(AS_native_pr, strand_behavior="same", join_type = "inner")
    dftop_outer_native_pr = dftop_outer_pr.join_overlaps(AS_native_pr, strand_behavior="same", join_type = "inner")


    dftop_inner_fixed_df = pl.from_pandas(dftop_inner_fixed_pr)
    dftop_inner_native_df = pl.from_pandas(dftop_inner_native_pr)

    def filter_outer(df: pl.DataFrame) -> pl.DataFrame:
        # Use flank length equal to N - M per request
        flank = max(N - M, 0)
        mask1 = (df["Start_b"] >= df["Start"]) & (df["Start_b"] <= df["Start"] + flank)
        mask2 = (df["Start_b"] >= df["End"] - flank) & (df["Start_b"] <= df["End"])  
        return df.filter(mask1 | mask2)

    dftop_outer_fixed_df = filter_outer(pl.from_pandas(dftop_outer_fixed_pr))
    dftop_outer_native_df = filter_outer(pl.from_pandas(dftop_outer_native_pr))

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
            .drop(['_wx', '_wx2'])
        )
        return res

    dftop_outer_fixed_sum = summarize_weighted_by_id(dftop_outer_fixed_df, value_col="AS_fixed", weight_col="depth_fixed")
    dftop_outer_native_sum = summarize_weighted_by_id(dftop_outer_native_df, value_col="AS_native", weight_col="depth_native")
    dftop_inner_fixed_sum = summarize_weighted_by_id(dftop_inner_fixed_df, value_col="AS_fixed", weight_col="depth_fixed")
    dftop_inner_native_sum = summarize_weighted_by_id(dftop_inner_native_df, value_col="AS_native", weight_col="depth_native")

    pwm_columns = ['inner_mean_logPWM', 'outer_mean_logPWM', 'GC_inner_pct', 'GC_outer_pct', 'kmer_count_inner', 'kmer_count_outer', 'diff_logPWM']

    fixed_part = dftop_outer_fixed_sum.join(dftop_inner_fixed_sum, on="id", how="inner", suffix="_inner")\
                           .drop([i+"_inner" for i in pwm_columns])\
                           .with_columns(
                               ((pl.col("AS_fixed_mean_inner") - pl.col("AS_fixed_mean")) /
                                (pl.col("AS_fixed_var_inner")/pl.col("_w_inner") + pl.col("AS_fixed_var")/pl.col("_w"))).alias("t_fixed")
                           )
    native_part = dftop_outer_native_sum.join(dftop_inner_native_sum, on="id", how="inner", suffix="_inner")\
                           .drop([i+"_inner" for i in pwm_columns])\
                           .with_columns(
                               ((pl.col("AS_native_mean_inner") - pl.col("AS_native_mean")) /
                                (pl.col("AS_native_var_inner")/pl.col("_w_inner") + pl.col("AS_native_var")/pl.col("_w"))).alias("t_native")
                           )

    results_df = fixed_part.join(native_part, on="id", suffix="_native")\
                            .with_columns((pl.col("t_native").pow(2) + pl.col("t_fixed").pow(2)).alias("s_l"))
    results_df.write_parquet(out_parquet)
def annotate_tss(
    input_parquet: str,
    out_parquet: str,
    rna_seq_parquet: str = "data/expression/ENCFF364YCB_HeLa_RNAseq_Transcripts_count_curated.parquet",
    phastcons_bed: str = "data/conservation_score/phastCons100way1.bed",
    phastcons_parquet: str = "data/conservation_score/hg38.phastCons100way.1.parquet",
    phylop_parquet: str = "data/conservation_score/hg38.phyloP100way.1.parquet",
    reference: str = "GRCh38",
):
    from metagene import load_reference, map_to_transcripts

    results_df = pl.read_parquet(input_parquet)
    RNA_seq_data = pl.read_parquet(rna_seq_parquet)
    conservative_df = pl.read_csv(phastcons_bed, separator="\t", has_header=False,
                                  new_columns=["Chromosome", "Start", "End", "Name"], comment_prefix="#")\
                         .with_columns(pl.col("Chromosome").str.replace('chr', '').alias("Chromosome"))

    ann_df = results_df.with_columns([
        pl.col("id").str.split("_").alias("id_split")
    ]).with_columns([
        pl.col("id_split").list.get(0).str.replace('chr', '').alias("Chromosome"),
        pl.col("id_split").list.get(2).alias("Strand"),
        pl.col("id_split").list.get(1).cast(pl.Int64).alias("Start")
    ]).drop("id_split").with_columns((pl.col("Start") + 50).alias("End"))\
      .with_columns((pl.col("Start") - 50).alias("Start"))

    reference_df = load_reference(reference)
    ann_df_tx = map_to_transcripts(ann_df, reference_df)

    ann_df_tx = ann_df_tx.with_columns((1 / (1 + pl.col("transcript_start")/1000)).fill_null(0).alias("TSS_proximity")
                                ).with_columns((1 / (1 + pl.col("transcript_start") / pl.col("transcript_length"))).fill_null(0).alias("Tx_percent"))
    conservative_pr = pr.PyRanges(conservative_df.to_pandas())
    ann_df_tx_pr = pr.PyRanges(ann_df_tx.to_pandas())

    joined_pr = ann_df_tx_pr.join_overlaps(conservative_pr, strand_behavior="ignore", join_type="left")
    joined_ann_df = pl.DataFrame(joined_pr)
    # percent of regions covered by PhastCons100way. 
    joined_ann_df = joined_ann_df.with_columns(
        (
            ((pl.min_horizontal("End", "End_b") - pl.max_horizontal("Start", "Start_b"))/100)
            .clip(lower_bound=0)
            .alias("PhastCons100_percent")
        )
    ).join(RNA_seq_data, on="transcript_id", how="left").with_columns(pl.col("TPM").fill_null(0))

    joined_ann_df = joined_ann_df.with_columns(pl.concat_str(pl.lit("chr"), pl.col("Chromosome")).alias("chrom"), (pl.col("Start") + 50).alias("start")).lazy()
    phastCons100df = pl.scan_parquet(phastcons_parquet)
    phylop100df = pl.scan_parquet(phylop_parquet)

    joined_ann_joined_df = joined_ann_df.sort(['chrom', 'start']).join_asof(phastCons100df, by="chrom", on="start", strategy="backward", suffix="_phastcons")\
                                        .filter(pl.col("start") <= pl.col("end"))\
                                        .join_asof(phylop100df, by="chrom", on="start", strategy="backward", suffix="_phylop100")\
                                        .filter(pl.col("start") <= pl.col("end_phylop100"))
    joined_ann_joined_df = joined_ann_joined_df.drop(["Chromosome",
                                                      "Start", "End", "Strand", "gene_id", "transcript_id", "transcript_start", "transcript_end",
                                                      "transcript_length", 'start_codon_pos', 'stop_codon_pos', 'exon_number', 'record_id', 'Start_exon',
                                                      'End_exon', 'Start_b', 'End_b', 'Name', 'end', 'end_phylop100',
                                                      'start', 'chrom'])\
                                           .rename({"score": "score_phastcons100"})
    joined_ann_joined_df.sink_parquet(out_parquet)


def annotate(
    input_parquet: Optional[str] = None,
    ASnative: Optional[str] = None,
    ASfixed: Optional[str] = None,
    toptable: Optional[str] = None,
    out_parquet: Optional[str] = None,
):
    os.makedirs("results", exist_ok=True)
    if input_parquet is None:
        if not (ASnative and ASfixed and toptable):
            raise SystemExit("Need input_parquet OR all of ASnative/ASfixed/toptable")
        tmp = (out_parquet or "results/annotated.parquet").replace("annotated", "sl")
        compute_sl(ASnative, ASfixed, toptable, tmp)
        input_parquet = tmp
    annotate_tss(input_parquet, out_parquet or "results/annotated.parquet")


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="accmix annotate")
    # annotate-acc parameters
    ap.add_argument("--ASnative", default=None, help="Accessibility native bed6 path")
    ap.add_argument("--ASfixed", default=None, help="Accessibility fixed bed6 path")
    ap.add_argument("--toptable", default=None, help="TopA PWM prior tsv.gz from accmix scan")
    ap.add_argument("--M", type=int, default=50, help="Inner half-width (default 50)")
    ap.add_argument("--N", type=int, default=500, help="Outer half-width (default 500)")
    ap.add_argument("--threshold", type=int, default=8, help="Minimum overlaps per id (default 8)")
    # annotate-tss parameters
    # WARNING: annotate-tss can use substantial memory on genome-scale inputs;
    # consider running on a machine with ample RAM or subsetting sites.
    ap.add_argument("--input-parquet", default=None, help="Parquet to annotate (e.g., output of annotate-acc)")
    ap.add_argument("--out", default=None, help="Output parquet path for annotated features")
    ap.add_argument("--rna-seq-parquet", default="data/expression/ENCFF364YCB_HeLa_RNAseq_Transcripts_count_curated.parquet")
    ap.add_argument("--phastcons-bed", default="data/conservation_score/phastCons100way1.bed")
    ap.add_argument("--phastcons-parquet", default="data/conservation_score/hg38.phastCons100way.1.parquet")
    ap.add_argument("--phylop-parquet", default="data/conservation_score/hg38.phyloP100way.1.parquet")
    ap.add_argument("--reference", default="GRCh38")
    return ap
