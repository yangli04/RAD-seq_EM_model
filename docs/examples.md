---
title: Examples
---

# Examples

End-to-end example using a small FASTA and example PWM.

```bash
# 1) Scan
accmix scan -f data/fasta/test.fa -p data/pwms/M00124_example.txt -o results/M00124_example

# 2) Compute s_l
accmix annotate-acc \
  -n data/AS/ANC1C.hisat3n_table.bed6 \
  -f data/AS/ANC1xC.hisat3n_table.bed6 \
  -t results/M00124_example_topA.tsv.gz \
  -o results/M00124_example_sl.parquet

# 3) Annotate TSS/conservation/TPM
accmix annotate-tss \
  -i results/M00124_example_sl.parquet \
  -o results/M00124_example_annotated.parquet \
  -r data/evaluation/RNAseq_HeLa_TPM.parquet \
  -c data/evaluation/phastCons100way.bed.gz \
  -p data/evaluation/phastCons100way.parquet \
  -y data/evaluation/phyloP100way.parquet \
  -R data/fasta/test.fa

# 4) Fit EM
accmix model -i results/M00124_example_annotated.parquet -o results/RBP_Motif -r ExampleRBP -m M00124
```
