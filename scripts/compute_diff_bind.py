import polars as pl
import glob, os
import pyranges as pr
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import seaborn as sns
from sklearn.metrics import roc_curve, auc
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import tqdm
import sys
sys.path.append('../scripts')
from model import run_em, read_polars_input, run_em_Gaussian
import em_utils


wt_df_files = sorted(glob.glob("../results/*_top_sl_table_annotated_TSS.parquet"))
wt_df_files_prefix = [i.replace("_top_sl_table_annotated_TSS.parquet", "") for i in wt_df_files]
ejckd_df_files = sorted(glob.glob("../results/*_top_sl_table_kd_annotated_TSS.parquet"))
ejckd_df_files_prefix = [i.replace("_top_sl_table_kd_annotated_TSS.parquet", "") for i in ejckd_df_files]

common_prefix = list(set(wt_df_files_prefix) & set(ejckd_df_files_prefix))
wt_df_files = [i + "_top_sl_table_annotated_TSS.parquet" for i in common_prefix]
ejckd_df_files = [i + "_top_sl_table_kd_annotated_TSS.parquet" for i in common_prefix]
motifs = [i.replace("../results/", "") for i in common_prefix]

rbp_all_info_filename = "../CISBP_RNA/RBP_Information_all_motifs.txt"
rbp_df = pl.read_csv(rbp_all_info_filename, separator="\t").select(['Motif_ID', 'RBP_Name'])

rbp_df = rbp_df.filter(
    pl.col("Motif_ID").is_in(motifs) 
).with_columns([
    # add the corresponding wt and ejckd files
    pl.col("Motif_ID").map_elements(lambda x: wt_df_files[motifs.index(x)], return_dtype=pl.Utf8).alias("wt_file"),
    pl.col("Motif_ID").map_elements(lambda x: ejckd_df_files[motifs.index(x)], return_dtype=pl.Utf8).alias("ejckd_file")
]).with_columns((pl.concat_str(pl.lit("/home/yangli/workspace/accessibility/CISBP_RNA/logos_all_motifs/"), pl.col("Motif_ID"), pl.lit("_fwd.png"))).alias("motif_logo"))

def run_em_and_intersect_pip(sample_filename):
    motif_df = pl.read_parquet(sample_filename)
    pipseq_pr = pr.PyRanges(pl.read_parquet("../metadatasets/PIPseq_HeLa.parquet").to_pandas())
    motif_range = 50
    motif_df = motif_df.with_columns([
        pl.col("id").str.split("_").alias("id_split")
    ]).with_columns([
        pl.col("id_split").list.get(0).alias("Chromosome"),
        pl.col("id_split").list.get(2).alias("Strand"),
        pl.col("id_split").list.get(1).cast(pl.Int64).alias("Position")
    ]).drop("id_split")


    motif_df = motif_df.with_columns([
        (pl.col("Position") - motif_range).alias("Start"),
        (pl.col("Position") + motif_range).alias("End")])

    motif_pandas = motif_df.to_pandas()
    motif_pr = pr.PyRanges(motif_pandas)

    pipintersections = motif_pr.intersect(pipseq_pr, strandedness = "same")
    pipintersected_ids = set(pipintersections.df['id'].tolist()) if len(pipintersections) > 0 else set()

    # Try
    motif_df = motif_df.with_columns([
        pl.when(pl.col("id").is_in(list(pipintersected_ids)))
        .then(pl.lit("pipseq"))
        .otherwise(pl.lit("non_determined"))
        .alias("source")
    ])
    motif_df = motif_df.drop(['Chromosome', 'Position', 'Start', 'End'])

    feature_columns = ['inner_mean_logPWM', 'outer_mean_logPWM', 'GC_inner_pct', 'GC_outer_pct',
                    'TSS_proximity', 'PhastCons100_percent', 'TPM', 'score_phastcons100', 'score_phylop100']
    skip_normalization = ['TSS_proximity', 'PhastCons100_percent', 'score_phastcons100']

    df, s, X, site_ids = read_polars_input(motif_df, "id", "s_l", feature_columns)
    X_zscore = X.copy()
    for i, col in enumerate(feature_columns):
        if col not in skip_normalization:
            col_zscore = stats.zscore(X[:, i])
            X_zscore[:, i] = np.nan_to_num(col_zscore, nan=0.0)
    # Make the first column to be all 1:
    X_zscore[:,0] = 1

    log_s = np.log(s + 1) 
    model, p, r = run_em_Gaussian(df, log_s, X_zscore)
    # model, p, r = run_em(df, s, X_zscore, tol = 1e-7)

    out_df = df.with_columns([
        pl.Series("prior_p", p),
        pl.Series("posterior_r", r),
    ])

    out_df = em_utils.change_label(out_df)
    return(out_df)

for sample_index in tqdm.tqdm(range(rbp_df.height)):
    score_phastcons100_threshold = 1.0
    wt_filename_i = rbp_df['wt_file'][sample_index]
    ejckd_filename_i = rbp_df['ejckd_file'][sample_index]
    motif_logo_i = rbp_df['motif_logo'][sample_index]

    file_title = f"{rbp_df['RBP_Name'][sample_index]}_{rbp_df['Motif_ID'][sample_index]}"
    print(file_title)

    wt_res = run_em_and_intersect_pip(wt_filename_i)
    ejckd_res = run_em_and_intersect_pip(ejckd_filename_i)

    # Add prediction categories to both datasets
    selected_columns = ['id', 'source', 'posterior_r', 'prior_p', 'predicted']
    wt_classified = wt_res.with_columns(
        pl.when(pl.col("posterior_r") > 0.99)
        .then(pl.lit(1))
        .when(pl.col("posterior_r") < 0.5)
        .then(pl.lit(0))
        .otherwise(pl.lit(2))
        .alias("predicted").cast(pl.Int8)
    ).select(selected_columns)

    ejckd_classified = ejckd_res.with_columns(
        pl.when(pl.col("posterior_r") > 0.99)
        .then(pl.lit(1))
        .when(pl.col("posterior_r") < 0.5)
        .then(pl.lit(0))
        .otherwise(pl.lit(2))
        .alias("predicted").cast(pl.Int8)
    ).select(selected_columns)

    # Join the two classified datasets with suffixes
    merged_df = wt_classified.join(
        ejckd_classified, 
        on="id", 
        how="inner",
        suffix="_ejckd"
    ).rename({
        "source": "source_wt",
        "posterior_r": "posterior_r_wt", 
        "prior_p": "prior_p_wt",
        "predicted": "predicted_wt"
    })

    # Create differential binding categories (0=unbound, 1=bound, 2=uncertain)
    merged_df = merged_df.with_columns([
        pl.when(
            (pl.col("predicted_wt") == 1) & (pl.col("predicted_ejckd") == 1)
        ).then(pl.lit("both_bound"))
        .when(
            (pl.col("predicted_wt") == 0) & (pl.col("predicted_ejckd") == 0)
        ).then(pl.lit("both_unbound"))
        .when(
            (pl.col("predicted_wt") == 1) & (pl.col("predicted_ejckd") == 0)
        ).then(pl.lit("wt_bound_ejckd_unbound"))
        .when(
            (pl.col("predicted_wt") == 0) & (pl.col("predicted_ejckd") == 1)
        ).then(pl.lit("wt_unbound_ejckd_bound"))
        .otherwise(pl.lit("uncertain"))
        .alias("differential_binding")
    ])

    merged_df.write_csv(f"../predicted_bound/{file_title}_differential_binding_results.tsv", separator = "\t")
