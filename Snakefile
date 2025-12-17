import glob, os
from os.path import basename, splitext, join, expanduser
import csv


# Original PWM configuration (commented out for TSS-only processing)
TRANSCRIPTS = expanduser("/home/yliuchicago/data/reference/Homo_sapiens/sequences/GCA_000001405.15_GRCh38_no_alt_analysis_set.chr22XYM.fna")
# PWM_DIR     = expanduser("/home/yliuchicago/workspace/accessibility/a_ruiqi_prevdata/centipede/CISBP/pwms_all_motifs")
# ACCDIR     = expanduser("/home/yliuchicago/workspace/accessibility/a_ruiqi_prevdata/preprocess/outdir/calculated_table/genome/")

# Discover motifs from precomputed AS-parquet tables
SCANNED_DIR = expanduser("/data/yangli/scanned_parquet")

# RBP/motif mapping and CLIP assets
RBP_INFO_PATH = expanduser("~/workspace/accessibility/CISBP_RNA/RBP_Information_all_motifs.txt")
CLIPSEQ_GLOB = "/home/yangli/train/add_rbp_clip/raw_data/*_HeLa.bed"
MOTIF_LOGO_DIR = "/home/yangli/workspace/accessibility/CISBP_RNA/logos_all_motifs"

def _discover_motifs(pattern: str, suffix: str):
    files = glob.glob(join(SCANNED_DIR, pattern))
    motifs = []
    for f in files:
        b = basename(f)
        if b.endswith(suffix):
            motifs.append(b.replace(suffix, ""))
    return sorted(set(motifs))

MOTIFS_NC = _discover_motifs("*_top_sl_table.parquet", "_top_sl_table.parquet")
MOTIFS_KD = _discover_motifs("*_top_sl_table_kd.parquet", "_top_sl_table_kd.parquet")
# MOTIFS_NC = ["M00437_2.00"]
# MOTIFS_KD = ["M00437_2.00"]

# Fallback/example when discovery yields nothing; comment out if not desired
MOTIFS = ["M00002"] if (not MOTIFS_NC and not MOTIFS_KD) else sorted(set(MOTIFS_NC) | set(MOTIFS_KD))

# Build motif -> RBP and RBP -> CLIP mappings
MOTIF2RBP = {}
if os.path.exists(os.path.expanduser(RBP_INFO_PATH)):
    try:
        with open(os.path.expanduser(RBP_INFO_PATH), newline="") as fh:
            rdr = csv.DictReader(fh, delimiter="\t")
            for row in rdr:
                motif_id = row.get("Motif_ID", "").strip()
                rbp_name = row.get("RBP_Name", "").strip()
                if motif_id and motif_id != "." and rbp_name:
                    MOTIF2RBP[motif_id] = rbp_name
    except Exception:
        MOTIF2RBP = {}

CLIPSEQ_FILES = sorted(glob.glob(CLIPSEQ_GLOB))
CLIPSEQ_SELECTED = [basename(i).replace("_HeLa.bed", "") for i in CLIPSEQ_FILES]
RBP2CLIP = {r: f for r, f in zip(CLIPSEQ_SELECTED, CLIPSEQ_FILES)}

def clip_for_motif_strict(motif: str):
    rbp = MOTIF2RBP.get(motif)
    if rbp and rbp in RBP2CLIP:
        return RBP2CLIP[rbp]
    raise ValueError(f"No CLIP bed found for motif {motif} (RBP: {rbp})")

def rbp_for_motif(motif: str):
    return MOTIF2RBP.get(motif, motif)

def logo_for_motif(motif: str):
    path = join(MOTIF_LOGO_DIR, f"{motif}_fwd.png")
    return path if os.path.exists(path) else ""

# Only evaluate motifs that have a CLIP bed mapping
MOTIFS_WITH_CLIP_NC = [m for m in MOTIFS_NC if MOTIF2RBP.get(m, None) in RBP2CLIP]
MOTIFS_WITH_CLIP_KD = [m for m in MOTIFS_KD if MOTIF2RBP.get(m, None) in RBP2CLIP]

# Filter motifs to only include those with existing parquet files
# EXISTING_MOTIFS = []
# for motif in MOTIFS:
#     base_file = f"results/{motif}_2.00_top_sl_table.parquet"
#     kd_file = f"results/{motif}_2.00_top_sl_table_kd.parquet"
#     if os.path.exists(base_file) and os.path.exists(kd_file):
#         EXISTING_MOTIFS.append(motif)

rule all:
    input:
        expand("results/{motif}_top_sl_table_nc_TSS.parquet", motif=MOTIFS_NC),
        expand("results/{motif}_top_sl_table_kd_TSS.parquet", motif=MOTIFS_KD),
        expand("results/{motif}_nc.model.json", motif=MOTIFS_NC),
        expand("results/{motif}_kd.model.json", motif=MOTIFS_KD),
        expand("results/{motif}_nc.model.parquet", motif=MOTIFS_NC),
        expand("results/{motif}_kd.model.parquet", motif=MOTIFS_KD),

        expand("results/{motif}_nc.evaluate.done", motif=MOTIFS_WITH_CLIP_NC),
        expand("results/{motif}_kd.evaluate.done", motif=MOTIFS_WITH_CLIP_KD)

# rule scan_pwm_topk:
#     input:
#         fasta = TRANSCRIPTS,
#         pwm   = "/project/mengjiechen/yliuchicago/workspace/accessibility/a_ruiqi_prevdata/centipede/CISBP/pwms_all_motifs/{motif}_2.00.txt"
#     output:
#         topA   = "results/{motif}_accmix_topA.tsv.gz",
#         botB   = temp("results/{motif}_accmix_botB.tsv.gz")
#     params:
#         out_prefix = lambda wc:f"results/{wc.motif}_accmix"
#     threads: 1
#     resources:
#         mem_mb=12000,
#     shell:
#         """
#         accmix scan -f {input.fasta} -p {input.pwm} --out-prefix "{params.out_prefix}"
#         """

# rule annt_sl_nc:
#     input:
#         topA   = "results/{motif}_accmix_topA.tsv.gz",
#         as_fixed = "/home/yliuchicago/workspace/accessibility/a_ruiqi_prevdata/preprocess/outdir/calculated_table/genome/ANC1C.hisat3n_table.bed6",
#         as_native = "/home/yliuchicago/workspace/accessibility/a_ruiqi_prevdata/preprocess/outdir/calculated_table/genome/ANC1xC.hisat3n_table.bed6"
#     output:
#         temp("results/{motif}_top_sl_table_nc.parquet"),
#     threads: 1
#     resources:
#         mem_mb=12000,
#     shell:
#         """
#         accmix annotate-acc -n {input.as_native} -f {input.as_fixed} -t {input.topA} -o {output}
#         """

# rule annt_sl_kd:
#     input:
#         topA   = "results/{motif}_accmix_topA.tsv.gz",
#         as_fixed = "/home/yliuchicago/workspace/accessibility/a_ruiqi_prevdata/preprocess/outdir/calculated_table/genome/AKD1C.hisat3n_table.bed6",
#         as_native = "/home/yliuchicago/workspace/accessibility/a_ruiqi_prevdata/preprocess/outdir/calculated_table/genome/AKD1xC.hisat3n_table.bed6"
#     output:
#         temp("results/{motif}_top_sl_table_kd.parquet"),
#     threads: 1
#     resources:
#         mem_mb=12000,
#     shell:
#         """
#         accmix annotate-acc -n {input.as_native} -f {input.as_fixed} -t {input.topA} -o {output}
#         """

rule annt_TSS_nc:
    input:
        lambda wc: join(SCANNED_DIR, f"{wc.motif}_top_sl_table.parquet"),
    output:
        "results/{motif}_top_sl_table_nc_TSS.parquet",
    threads: 8
    resources:
        mem_mb=30000,
    shell:
        """
        accmix annotate-tss -i {input} -o {output} -r data/expression/ENCFF364YCB_HeLa_RNAseq_Transcripts_count_curated.parquet -c data/conservation_score/phastCons100way1.bed -p data/conservation_score/hg38.phastCons100way.1.parquet -y data/conservation_score/hg38.phyloP100way.1.parquet
        """

rule annt_TSS_kd:
    input:
        lambda wc: join(SCANNED_DIR, f"{wc.motif}_top_sl_table_kd.parquet"),
    output:
        "results/{motif}_top_sl_table_kd_TSS.parquet",
    threads: 8
    resources:
        mem_mb=30000,
    shell:
        """
        accmix annotate-tss -i {input} -o {output} -r data/expression/ENCFF364YCB_HeLa_RNAseq_Transcripts_count_curated.parquet -c data/conservation_score/phastCons100way1.bed -p data/conservation_score/hg38.phastCons100way.1.parquet -y data/conservation_score/hg38.phyloP100way.1.parquet
        """

# --- Model fitting ---

rule model_nc:
    input:
        ann = "results/{motif}_top_sl_table_nc_TSS.parquet",
    output:
        json = "results/{motif}_nc.model.json",
        parquet = "results/{motif}_nc.model.parquet",
    threads: 2
    resources:
        mem_mb=10000,
    shell:
        """
        accmix model -i {input.ann} -o results/{wildcards.motif}_nc -r {wildcards.motif} -m {wildcards.motif}
        """

rule model_kd:
    input:
        ann = "results/{motif}_top_sl_table_kd_TSS.parquet",
    output:
        json = "results/{motif}_kd.model.json",
        parquet = "results/{motif}_kd.model.parquet",
    threads: 2
    resources:
        mem_mb=10000,
    shell:
        """
        accmix model -i {input.ann} -o results/{wildcards.motif}_kd -r {wildcards.motif} -m {wildcards.motif}
        """

# --- Evaluation ---

rule evaluate_nc:
    input:
        model = "results/{motif}_nc.model.parquet",
        clip = lambda wc: clip_for_motif_strict(wc.motif),
        pipseq = "data/evaluation/PIPseq_HeLa.parquet",
    output:
        touch("results/{motif}_nc.evaluate.done"),
    threads: 1
    resources:
        mem_mb=12000,
    params:
        rbp=lambda wc: rbp_for_motif(wc.motif),
        logo=lambda wc: logo_for_motif(wc.motif),
    shell:
        """
        accmix evaluate \
            -M {input.model} \
            -b {input.clip} \
            -p {input.pipseq} \
            -r {params.rbp} \
            -m {wildcards.motif} \
            -o results \
            -L {params.logo} \
            -t 1.0 \
            -R 50 \
        && touch {output}
        """

rule evaluate_kd:
    input:
        model = "results/{motif}_kd.model.parquet",
        clip = lambda wc: clip_for_motif_strict(wc.motif),
        pipseq = "data/evaluation/PIPseq_HeLa.parquet",
    output:
        touch("results/{motif}_kd.evaluate.done"),
    threads: 1
    resources:
        mem_mb=12000,
    params:
        rbp=lambda wc: rbp_for_motif(wc.motif),
        logo=lambda wc: logo_for_motif(wc.motif),
    shell:
        """
        accmix evaluate \
            -M {input.model} \
            -b {input.clip} \
            -p {input.pipseq} \
            -r {params.rbp} \
            -m {wildcards.motif} \
            -o results \
            -L {params.logo} \
            -t 1.0 \
            -R 50 \
        && touch {output}
        """
