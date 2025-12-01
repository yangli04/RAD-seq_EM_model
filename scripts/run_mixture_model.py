import polars as pl
import glob, os
import pyranges as pr
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import seaborn as sns
from sklearn.metrics import roc_curve, auc
from sklearn.linear_model import LogisticRegression
import sys
sys.path.append('../scripts')
from model import run_em, read_polars_input, run_em_Gaussian
import em_utils

score_phastcons100_threshold = 1.0
motif_range = 50
motif_files = sorted(glob.glob("../midway3_results/M*_wtTSS_and*.parquet"))
motif_selected = [os.path.basename(f).replace("_wtTSS_and_PhaseCons.parquet", "") for f in motif_files]
clipseq_files = sorted(glob.glob("/home/yangli/train/add_rbp_clip/raw_data/*_HeLa.bed")) 
clipseq_selected = [os.path.basename(i).replace("_HeLa.bed", "") for i in clipseq_files]
rbp_all_info_filename = "~/workspace/accessibility/CISBP_RNA/RBP_Information_all_motifs.txt"
rbp_df = pl.read_csv(rbp_all_info_filename, separator="\t").select(['Motif_ID', 'RBP_Name'])

motif_map = {m: f for m, f in zip(motif_selected, motif_files)}
clipseq_map = {c: f for c, f in zip(clipseq_selected, clipseq_files)}

rbp_df = rbp_df.filter(
    pl.col("Motif_ID").is_in(motif_selected) & pl.col("RBP_Name").is_in(clipseq_selected)
).with_columns([
    pl.col("Motif_ID").map_elements(motif_map.get, return_dtype = str).alias("motif_file"),
    pl.col("RBP_Name").map_elements(clipseq_map.get, return_dtype = str).alias("clipseq_file")
]).with_columns((pl.concat_str(pl.lit("/home/yangli/workspace/accessibility/CISBP_RNA/logos_all_motifs/"), pl.col("Motif_ID"), pl.lit("_fwd.png"))).alias("motif_logo"))


for sample_index in range(len(rbp_df)):
    # sample_index = 26
    try:
        motif_file_i = rbp_df['motif_file'][sample_index]
        clipseq_file_i = rbp_df['clipseq_file'][sample_index]
        motif_logo_i = rbp_df['motif_logo'][sample_index]
        rbp_df['Motif_ID'][sample_index]
        rbp_df['RBP_Name'][sample_index]
        file_title = f"{rbp_df['RBP_Name'][sample_index]}_{rbp_df['Motif_ID'][sample_index]}"
        print(f"dealing with {file_title}")

        motif_df = pl.read_parquet(motif_file_i)
        clipseq_df = pl.read_csv(clipseq_file_i, separator="\t", new_columns = ['Chromosome', 'Start', 'End', 'peak_id', 'score', 'Strand', 'RBP_Name', 'method', 'cell_line', 'datasource'])
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
        clipseq_pandas = clipseq_df.select([
            "Chromosome", "Start", "End", "Strand", "peak_id", "score"
        ]).to_pandas()
        clipseq_pr = pr.PyRanges(clipseq_pandas)

        # pip_seq_data = "~/train/add_rbp_clip/raw_data/RBP_occupancy_HeLa.bed"
        # pl.read_csv(pip_seq_data, separator="\t", has_header = False, new_columns = ['Chromosome', 'Start', "End", "peak_name", "score", 'Strand', "exp", 'Type', "cell", "from"]).select(['Chromosome', 'Start', "End", 'Strand', "score"]).write_parquet("../results/PIPseq_HeLa.parquet")
        pipseq_pr = pr.PyRanges(pl.read_parquet("../results/PIPseq_HeLa.parquet").to_pandas())

        intersections = motif_pr.intersect(clipseq_pr, strandedness="same")
        pipintersections = motif_pr.intersect(pipseq_pr, strandedness = "same")
        intersected_ids = set(intersections.df['id'].tolist()) if len(intersections) > 0 else set()
        pipintersected_ids = set(pipintersections.df['id'].tolist()) if len(pipintersections) > 0 else set()

        # Try
        motif_df = motif_df.with_columns([
            pl.when(pl.col("id").is_in(list(intersected_ids)))
            .then(pl.lit("clip_bound"))
            .when(
                ~pl.col("id").is_in(list(pipintersected_ids)) & 
                (pl.col("score_phastcons100") <= score_phastcons100_threshold)
            )
            .then(pl.lit("clip_unbound"))
            .otherwise(pl.lit("non_determined"))
            .alias("source"),
            # (1 / (1 + (1 / (pl.col("TSS_proximity") + 10**(-10)) - 1) / 1000)).alias("TSS_proximity")
        ])


        motif_df = motif_df.drop(['Chromosome', 'Position', 'Start', 'End'])
        # TSS proximity is using 1 / (1 + TSS distance / 1000).
        # Does not perform better...
        feature_columns = ['inner_mean_logPWM', 'outer_mean_logPWM', 'GC_inner_pct', 'GC_outer_pct',
                        'TSS_proximity', 'PhastCons100_percent', 'TPM', 'score_phastcons100', 'score_phylop100']
        skip_normalization = ['TSS_proximity', 'PhastCons100_percent', 'score_phastcons100']

        df, s, X, site_ids = read_polars_input(motif_df, "id", "s_l", 
                                feature_columns)
        X_zscore = X.copy()
        for i, col in enumerate(feature_columns):
            if col not in skip_normalization:
                col_zscore = stats.zscore(X[:, i])
                X_zscore[:, i] = np.nan_to_num(col_zscore, nan=0.0)
        # Make the first column to be all 1:
        X_zscore[:,0] = 1

        log_s = np.log(s + 1) 
        # model, p, r = run_em(df, log_s, X_zscore)
        model, p, r = run_em_Gaussian(df, log_s, X_zscore)
        # model, p, r = run_em(df, s, X_zscore)
        out_df = df.with_columns([
            pl.Series("prior_p", p),
            pl.Series("posterior_r", r),
        ])
        out_df = em_utils.change_label(out_df)
        plot_out_df = out_df.group_by('source').agg(pl.col('posterior_r').mean().alias("posterior_r_mean"), 
                                    pl.col('prior_p').mean().alias("prior_p_mean"))

        roc_auc = em_utils.plot_auc_em(out_df, file_title, save_path = None)
        roc_auc_control = em_utils.plot_auc_em_control(out_df, file_title, save_path = None)
        expanded_feature_columns = feature_columns + ['prior_p', 'posterior_r', 's_l', 'source']
        em_utils.heatmap_plot(data = out_df, columns = expanded_feature_columns, 
                            no_normalize_columns = skip_normalization + ['source'], 
                            save_heatmap_path=f"../plots/{file_title}_heatmap_with_title_logo.png", file_title = file_title, motif_logo_i = motif_logo_i, rbp_df = rbp_df, sample_index = sample_index)

        test_dict = em_utils.compute_QQ_validate_distribution_robust(out_df, save_figure_path=f"../plots/distribution/{file_title}.dist.val.png")
        em_utils.plot_beta_coefficients(model, feature_columns, file_title)
        # em_utils.plot_gamma_distribution_fit(out_df, model, file_title)
        print("=== Logistic Regression with 'source' labels ===")

        source_mask = out_df['source'] != 'non_determined'
        X_source = X_zscore[source_mask.to_numpy(), 1:]
        y_source_labels = out_df.filter(source_mask)['source'].to_numpy()

        # Convert to binary: clip_bound=1, clip_unbound=0
        y_source = (y_source_labels == 'clip_bound').astype(int)

        # Fit logistic regression
        lr_source = LogisticRegression(max_iter=1000, random_state=42)
        lr_source.fit(X_source, y_source)

        # Get coefficients and predictions
        source_coefs = lr_source.coef_[0]
        source_intercept = lr_source.intercept_[0]
        y_source_proba = lr_source.predict_proba(X_source)[:, 1]

        # Calculate ROC
        fpr_source, tpr_source, _ = roc_curve(y_source, y_source_proba)
        auc_source = auc(fpr_source, tpr_source)

        X_zscore_s_l = X_zscore.copy()
        # Insert the s_l zscore to the first column. 
        X_zscore_s_l[:, 0] = stats.zscore(log_s)
        X_source_s_l = X_zscore_s_l[source_mask.to_numpy(), ]
        lr_source_s_l = LogisticRegression(max_iter=1000, random_state=42)
        lr_source_s_l.fit(X_source_s_l, y_source)

        # Get coefficients and predictions
        source_coefs_s_l = lr_source_s_l.coef_[0]
        source_intercept_s_l = lr_source_s_l.intercept_[0]
        y_source_proba_s_l = lr_source_s_l.predict_proba(X_source_s_l)[:, 1]

        # Calculate ROC
        fpr_source_s_l, tpr_source_s_l, _ = roc_curve(y_source, y_source_proba_s_l)
        auc_source_s_l = auc(fpr_source_s_l, tpr_source_s_l)

        # beta_results_df_s_l, beta_results_df, metadata_df = em_utils.save_analysis_results(model, source_coefs, source_coefs_s_l, source_intercept, source_intercept_s_l, feature_columns, roc_auc, auc_source, auc_source_s_l,
                            # file_title, out_df, test_dict = test_dict, output_dir='../logs')
                                                        # ...existing code...
        beta_results_df_s_l, beta_results_df, metadata_df = em_utils.save_analysis_results(
            model, source_coefs, source_coefs_s_l, source_intercept, source_intercept_s_l,
            feature_columns, roc_auc, roc_auc_control, auc_source, auc_source_s_l,
            file_title, out_df, test_dict=test_dict, motif_df=motif_df, output_dir='../logs'
        )
    except Exception as e:
        print(f"{e}")
