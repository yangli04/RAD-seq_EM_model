# Please run this code in metagene environment, at least install the metagene package. Otherwise the current version of pyranges do not support it as it used some outdated pyranges function. 
import polars as pl
from metagene import load_sites, load_reference, map_to_transcripts
import pyranges as pr
import sys
import glob

# parquet_files = sorted(glob.glob("../results/M*_top_sl_table.parquet"))
# parquet_files = sorted(glob.glob("../midway3_results/M*_top_sl_table.parquet"))

# for sample_index in range(len(parquet_files)):
    # file_name = parquet_files[sample_index]
    # save_file_name = parquet_files[sample_index].replace("_top_sl_table.parquet", "_wtTSS_and_PhaseCons.parquet")
file_name = sys.argv[1]
save_file_name = sys.argv[2]
# save_file_name = file_name.replace("_top_sl_table.parquet", "_wtTSS_and_PhaseCons.parquet")

results_df = pl.read_parquet(file_name)
RNA_seq_data = pl.read_parquet("../metadatasets/ENCFF364YCB_HeLa_RNAseq_Transcripts_count_curated.parquet")
conservative_df = pl.read_csv("../metadatasets/phastCons100way1.bed", separator = "\t", has_header = False, 
                            new_columns = ["Chromosome", "Start", "End", "Name"], comment_prefix="#").with_columns(
                                pl.col("Chromosome").str.replace('chr', '').alias("Chromosome"))

# Expand to 100 bp and calculate the overlapped region / 100 as the conservative score.
ann_df = results_df.with_columns([
    pl.col("id").str.split("_").alias("id_split")
]).with_columns([
    pl.col("id_split").list.get(0).str.replace('chr', '').alias("Chromosome"),
    pl.col("id_split").list.get(2).alias("Strand"),
    pl.col("id_split").list.get(1).cast(pl.Int64).alias("Start")
]).drop("id_split").with_columns((pl.col("Start") + 50).alias("End")).with_columns((pl.col("Start") - 50).alias("Start"))

reference_df = load_reference("GRCh38")
ann_df_tx = map_to_transcripts(ann_df, reference_df)
print("mapped to transcripts!")
# include TSS proximity score. 1/ (1 + transcript_start), null values are assigned 0.
# calculate another called tx_percent score, the relative position within the transcript, by taking the position divided by the transcript length

# This is the previous version without scaling, not aligned for CENTIPEDE model 
# ann_df_tx = ann_df_tx.with_columns((1 / (1 + pl.col("transcript_start"))).fill_null(0).alias("TSS_proximity")

# This is the updated version with scaling (./1000), same as CENTIPEDE model
ann_df_tx = ann_df_tx.with_columns((1 / (1 + pl.col("transcript_start")/1000)).fill_null(0).alias("TSS_proximity")
                                ).with_columns((1 / (1 + pl.col("transcript_start") / pl.col("transcript_length"))).fill_null(0).alias("Tx_percent"))
conservative_pr = pr.PyRanges(conservative_df.to_pandas())
ann_df_tx_pr = pr.PyRanges(ann_df_tx.to_pandas())
joined_ann_df = pl.DataFrame(ann_df_tx_pr.join_overlaps(conservative_pr, strand_behavior = "ignore", join_type = "left"))
joined_ann_df = joined_ann_df.with_columns(
                    (
                        ((pl.min_horizontal("End", "End_b") - pl.max_horizontal("Start", "Start_b"))/ 100)
                        .clip(lower_bound=0)  # ensures no negative lengths
                        .alias("PhastCons100_percent")
                    )).join(RNA_seq_data, on = "transcript_id", how = "left").with_columns(pl.col("TPM").fill_null(0)
                    )

joined_ann_df = joined_ann_df.with_columns(pl.concat_str(pl.lit("chr"), pl.col("Chromosome")).alias("chrom"), (pl.col("Start") + 50).alias("start")).lazy()
phastCons100df = pl.scan_parquet("../midway3_results/hg38.phastCons100way.1.parquet")
phylop100df = pl.scan_parquet("../midway3_results/hg38.phyloP100way.1.parquet")

joined_ann_joined_df = joined_ann_df.sort(['chrom', 'start']).join_asof(phastCons100df, by = "chrom", on = "start", strategy="backward", suffix = "_phastcons"
                                        ).filter(pl.col("start") <= pl.col("end")
                                        ).join_asof(phylop100df, by = "chrom", on = "start", strategy="backward", suffix = "_phylop100"
                                        ).filter(pl.col("start") <= pl.col("end_phylop100"))
joined_ann_joined_df = joined_ann_joined_df.drop(["Chromosome", 
                                                "Start", "End", "Strand", "gene_id", "transcript_id", "transcript_start", "transcript_end", 
                                                "transcript_length", 'start_codon_pos', 'stop_codon_pos', 'exon_number', "record_id", "Start_exon", 
                                                "End_exon", "Start_b", "End_b", "Name", "end", "end_phylop100", 
                                                "start", "chrom"]
                                                ).rename({"score": "score_phastcons100"})
joined_ann_joined_df.sink_parquet(save_file_name)
