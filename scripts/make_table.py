import polars as pl
import pyranges as pr
import argparse


parser = argparse.ArgumentParser(
    description = "Take accessibility adata and the pwm prior data to calculate sl score"
)
parser.add_argument("-n", "--ASnative", type = str, default = "/home/yangli/workspace/accessibility/calculated_table/genome/ANC1C.hisat3n_table.bed6", help = "AS native")
parser.add_argument("-f", "--ASfixed", type = str, default = "/home/yangli/workspace/accessibility/calculated_table/genome/ANC1xC.hisat3n_table.bed6", help = "AS fixed")
parser.add_argument("-t", "--toptable", type = str, default = "../results/M00124_2.00_topA.tsv.gz", help = "toptable, computed prior")
parser.add_argument("-o", "--output", type = str, default = "../results/test.parquet", help = "output parquet")

args = parser.parse_args()


AS_fixed = pl.read_csv(args.ASfixed, has_header = False, new_columns = ["Chromosome", "Start", "Strand", "AS_fixed", 'depth_fixed', "motif"], separator = "\t").select(["Chromosome", "Start", "Strand", "AS_fixed", 'depth_fixed']).with_columns((pl.col("Start") + 1).alias("End"))
AS_native = pl.read_csv(args.ASnative, has_header = False, new_columns = ["Chromosome", "Start", "Strand", "AS_native", 'depth_native', "motif"], separator = "\t").select(["Chromosome", "Start", "Strand", "AS_native", 'depth_native']).with_columns((pl.col("Start") + 1).alias("End"))
AS_fixed_pr = pr.PyRanges(AS_fixed.to_pandas())
AS_native_pr = pr.PyRanges(AS_native.to_pandas())
dftop = pl.read_csv(args.toptable, separator="\t").with_columns((pl.col("inner_mean_logPWM") - pl.col("outer_mean_logPWM")).alias("diff_logPWM")).sort(['chrom', 'pos1', 'strand']).with_columns(pl.concat_str(['chrom', 'pos1', 'strand'], separator="_").alias("id"))
output_parquet_file = args.output
# inner and outer region defination
M = 50 
N = 500
dftop_inner = dftop.with_columns((pl.col("pos1") - pl.lit(M)).alias("Start"), (pl.col("pos1") + pl.lit(M)).alias("End")).rename({"chrom": "Chromosome", "strand": "Strand"})
dftop_outer = dftop.with_columns((pl.col("pos1") - pl.lit(N)).alias("Start"), (pl.col("pos1") + pl.lit(N)).alias("End")).rename({"chrom": "Chromosome", "strand": "Strand"})
dftop_inner_pr = pr.PyRanges(dftop_inner.to_pandas())
dftop_outer_pr = pr.PyRanges(dftop_outer.to_pandas())


# Join with strand consideration - use "same" for same strand overlaps
dftop_inner_fixed_pr = dftop_inner_pr.join(AS_fixed_pr, strandedness="same")
dftop_outer_fixed_pr = dftop_outer_pr.join(AS_fixed_pr, strandedness="same")
dftop_inner_native_pr = dftop_inner_pr.join(AS_native_pr, strandedness="same")
dftop_outer_native_pr = dftop_outer_pr.join(AS_native_pr, strandedness="same")



dftop_inner_fixed_df = pl.from_pandas(dftop_inner_fixed_pr.df)
dftop_inner_native_df = pl.from_pandas(dftop_inner_native_pr.df)


def filter_outer(df):
    mask1 = (df["Start_b"] >= df["Start"]) & (df["Start_b"] <= df["Start"] + 450)
    mask2 = (df["Start_b"] >= df["End"] - 450) & (df["Start_b"] <= df["End"])
    return df.filter(mask1 | mask2)

dftop_outer_fixed_df = filter_outer(pl.from_pandas(dftop_outer_fixed_pr.df))
dftop_outer_native_df = filter_outer(pl.from_pandas(dftop_outer_native_pr.df))


def summarize_weighted_by_id(
    df: pl.DataFrame,
    value_col: str,
    weight_col: str,
    *,
    id_col: str = "id",
    threshold: int = 8,
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
        # .select([id_col, "n", f"{value_col}_mean", f"{value_col}_var", w_mean_name, f"{value_col}_w_var"])
    )
    return res


dftop_outer_fixed_sum = summarize_weighted_by_id(dftop_outer_fixed_df, value_col = "AS_fixed", weight_col = "depth_fixed")
dftop_outer_native_sum = summarize_weighted_by_id(dftop_outer_native_df, value_col = "AS_native", weight_col = "depth_native")
dftop_inner_fixed_sum = summarize_weighted_by_id(dftop_inner_fixed_df, value_col = "AS_fixed", weight_col = "depth_fixed")
dftop_inner_native_sum = summarize_weighted_by_id(dftop_inner_native_df, value_col = "AS_native", weight_col = "depth_native")


pwm_columns = ['inner_mean_logPWM', 'outer_mean_logPWM', 'GC_inner_pct', 'GC_outer_pct', 'kmer_count_inner', 'kmer_count_outer', 'diff_logPWM']

fixed_part = dftop_outer_fixed_sum.join(dftop_inner_fixed_sum, on = "id", how = "inner", suffix = "_inner"
                           ).drop([i+"_inner" for i in pwm_columns]
                           ).with_columns(
                               ((pl.col("AS_fixed_mean_inner") - pl.col("AS_fixed_mean")) / (pl.col("AS_fixed_var_inner")/pl.col("_w_inner") + pl.col("AS_fixed_var")/pl.col("_w"))).alias("t_fixed")
                           )
native_part = dftop_outer_native_sum.join(dftop_inner_native_sum, on = "id", how = "inner", suffix = "_inner"
                           ).drop([i+"_inner" for i in pwm_columns]
                           ).with_columns(
                               ((pl.col("AS_native_mean_inner") - pl.col("AS_native_mean")) / (pl.col("AS_native_var_inner")/pl.col("_w_inner") + pl.col("AS_native_var")/pl.col("_w"))).alias("t_native")
                           )

results_df = fixed_part.join(native_part, on = "id", suffix = "_native"
                            ).with_columns(
                                (pl.col("t_native").pow(2) + pl.col("t_fixed").pow(2)).alias("s_l")
                            )
results_df.write_parquet(output_parquet_file)
