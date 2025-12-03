import glob, os
from os.path import basename, splitext, join, expanduser

TRANSCRIPTS = "data/fasta/genome.small.fa.gz"
PWM_DIR     = "data/pwms"

PWM_TXTS = sorted(glob.glob(join(PWM_DIR, "*.txt")))
MOTIFS   = [splitext(basename(p))[0] for p in PWM_TXTS]
MOTIF2PWM = dict(zip(MOTIFS, PWM_TXTS))

rule all:
    input:
        expand("results/{motif}_topA.tsv.gz", motif=MOTIFS)

rule scan_pwm_topk:
    input:
        fasta = TRANSCRIPTS,
        pwm   = lambda wc: MOTIF2PWM[wc.motif]
    output:
        topA   = "results/{motif}_topA.tsv.gz",
        botB   = "results/{motif}_botB.tsv.gz"
    params:
        scan_pwm_regions = "scripts/scan_pwm_regions.py",
        out_prefix = lambda wc: f"results/{wc.motif}"
    threads: 1
    shell:
        """
        python {params.scan_pwm_regions} --fasta {input.fasta} --pwm {input.pwm} \
        -M 50 -N 500 -A 3000000 -B 3000000 --bg "0.26650202,0.23772517,0.2433633,0.2524095" --out-prefix {params.out_prefix}
        """
