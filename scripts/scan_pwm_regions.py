#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Strand-separated, reverse-complement–exact genome regional PWM + GC scanner (LLR-ready).

We compute strand-specific region means by scanning TWO sequences per contig:
  (+) the original contig
  (−) its reverse complement (as if it were a separate FASTA)

For each center position i:
  Inner region: [i-M, i+M]
  Outer regions: [i-N, i-M-1] U [i+M+1, i+N]
Region score = mean of k-mer scores whose starts are fully inside the region.

Scoring:
  - Per k-mer score = sum_k log PWM prob (or LLR if background q provided).
  - Only a forward lookup table is needed; the minus strand is handled
    by actually scoring the reverse-complement sequence.

Outputs:
  <prefix>_topA.tsv.gz, <prefix>_botB.tsv.gz with strand-aware records.

Columns:
  chrom pos1 strand inner_mean_logPWM outer_mean_logPWM
  GC_inner_pct GC_outer_pct kmer_count_inner kmer_count_outer
"""
from __future__ import annotations
import argparse, gzip, sys, heapq
from typing import Tuple, List, Iterable, Optional
import numpy as np

# -------- Optional JIT --------
try:
    from numba import njit
    NUMBA_OK = True
except Exception:
    NUMBA_OK = False

# -------- FASTA I/O --------
def open_maybe_gzip(path: str):
    return gzip.open(path, "rt") if path.endswith(".gz") else open(path, "rt")

def fasta_records(path: str) -> Iterable[Tuple[str, str]]:
    with open_maybe_gzip(path) as fh:
        name = None
        buf = []
        for line in fh:
            if not line:
                break
            if line[0] == '>':
                if name is not None:
                    yield name, ''.join(buf).upper()
                name = line[1:].strip().split()[0]
                buf = []
            else:
                buf.append(line.strip())
        if name is not None:
            yield name, ''.join(buf).upper()

# -------- PWM & background --------
def read_pwm_txt(path: str) -> np.ndarray:
    """Return logP[L,4] (natural log) in A,C,G,T order (accepts U)."""
    with open(path, "r") as fh:
        lines = [ln.strip() for ln in fh if ln.strip()]
    header = lines[0].split()
    col = {h.upper(): i for i, h in enumerate(header)}
    if "U" in col and "T" in col:
        raise ValueError("PWM header cannot contain both U and T.")
    if "U" in col:
        col["T"] = col["U"]
    need = ["POS", "A", "C", "G", "T"]
    for k in need:
        if k not in col:
            raise ValueError(f"Missing column '{k}' in PWM header: {header}")
    data = []
    for ln in lines[1:]:
        toks = ln.split()
        A = float(toks[col["A"]]); C = float(toks[col["C"]])
        G = float(toks[col["G"]]); T = float(toks[col["T"]])
        data.append([A, C, G, T])
    pwm = np.array(data, dtype=np.float64)  # L x 4
    eps = 1e-9
    pwm = pwm + eps
    pwm = pwm / pwm.sum(axis=1, keepdims=True)
    return np.log(pwm)

def parse_bg_arg(s: str) -> np.ndarray:
    vals = [float(x) for x in s.split(",")]
    if len(vals) != 4:
        raise ValueError("Expected 4 comma-separated probs for --bg (A,C,G,T).")
    arr = np.array(vals, dtype=np.float64)
    if not np.isfinite(arr).all() or (arr <= 0).any():
        raise ValueError("Background probs must be positive & finite.")
    arr /= arr.sum()
    return arr

def estimate_bg_from_fasta(fasta_path: str) -> np.ndarray:
    cnt = np.zeros(4, dtype=np.int64)
    for _, seq in fasta_records(fasta_path):
        a = np.frombuffer(seq.encode('ascii'), dtype=np.uint8)
        cnt[0] += np.count_nonzero(a == 65)                  # A
        cnt[1] += np.count_nonzero(a == 67)                  # C
        cnt[2] += np.count_nonzero(a == 71)                  # G
        cnt[3] += np.count_nonzero((a == 84) | (a == 85))    # T/U
    q = cnt.astype(np.float64)
    q = np.where(q > 0, q, 1.0)
    q /= q.sum()
    return q

def apply_background_llr(logp_rows: np.ndarray, q: Optional[np.ndarray]) -> np.ndarray:
    """Return per-position log-odds rows: logp - logq if q provided, else logp."""
    if q is None:
        return logp_rows
    logq = np.log(q)
    return logp_rows - logq[None, :]

# -------- K-mer lookup (forward only) --------
def build_kmer_table_fwd(logp: np.ndarray) -> np.ndarray:
    """fwd_table[code] = log-score of k-mer in forward orientation."""
    L = logp.shape[0]
    K = 4 ** L
    fwd = np.empty(K, dtype=np.float32)
    for code in range(K):
        s = 0.0
        c = code
        for pos in range(L):
            b = c & 3
            s += logp[L - 1 - pos, b]
            c >>= 2
        fwd[code] = s
    return fwd

# -------- Numba kernels --------
if NUMBA_OK:
    @njit(cache=True, fastmath=True)
    def score_kmers_forward(seq_codes: np.ndarray, L: int, table_fwd: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return (scores_fwd[t], valid[t]) for k-mer starting at t on the given sequence.
        """
        n = seq_codes.size
        T = n - L + 1
        sf = np.zeros(T, dtype=np.float32)
        valid = np.zeros(T, dtype=np.uint8)
        if T <= 0:
            return sf, valid
        mask = (1 << (2*L)) - 1
        code = 0
        valid_len = 0
        # Compute kmer code in rolling fashion. When we hit an invalid base, reset.
        for i in range(n):
            b = seq_codes[i]
            if b < 4:
                code = ((code << 2) & mask) | b
                valid_len += 1
            else:
                code = 0
                valid_len = 0
            if valid_len >= L:
                t = i - L + 1
                sf[t] = table_fwd[code]
                valid[t] = 1
        return sf, valid

    @njit(cache=True, fastmath=True)
    def cumsum_int8_to_int32(x: np.ndarray) -> np.ndarray:
        out = np.empty(x.size + 1, dtype=np.int32)
        s = 0
        out[0] = 0
        for i in range(x.size):
            s += int(x[i])
            out[i+1] = s
        return out

    @njit(cache=True, fastmath=True)
    def cumsum_f32_to_f64(x: np.ndarray) -> np.ndarray:
        out = np.empty(x.size + 1, dtype=np.float64)
        s = 0.0
        out[0] = 0.0
        for i in range(x.size):
            s += float(x[i])
            out[i+1] = s
        return out

# -------- helpers --------
def base_to_code_array(seq: str) -> np.ndarray:
    m = np.full(len(seq), 255, dtype=np.uint8)
    a = np.frombuffer(seq.encode('ascii'), dtype=np.uint8, count=len(seq))
    m[(a == 65)] = 0                     # A
    m[(a == 67)] = 1                     # C
    m[(a == 71)] = 2                     # G
    m[(a == 84) | (a == 85)] = 3         # T/U
    return m

def gc_bool_array(seq: str) -> np.ndarray:
    a = np.frombuffer(seq.encode('ascii'), dtype=np.uint8, count=len(seq))
    return ((a == 67) | (a == 71)).astype(np.uint8)  # C/G

def reverse_complement(seq: str) -> str:
    tbl = str.maketrans("ACGTUacgtu", "TGCAAugcaa")
    # Replace U->T first so we keep DNA alphabet consistent:
    seq = seq.replace('U', 'T').replace('u', 't')
    return seq.translate(str.maketrans("ACGTacgt", "TGCAtgca"))[::-1]

def fallback_score_forward(seq: str, logp: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Slow O(L*N) scorer if Numba is unavailable."""
    print("[warn] Numba not available; using slow scorer.", file=sys.stderr)
    L = logp.shape[0]
    n = len(seq)
    T = n - L + 1
    sf = np.zeros(max(T, 0), dtype=np.float32)
    valid = np.zeros(max(T, 0), dtype=np.uint8)
    for t in range(T):
        kmer = seq[t:t+L]
        if any(c not in "ACGT" for c in kmer):
            continue
        s1 = 0.0
        for k, ch in enumerate(kmer):
            b = "ACGT".index(ch)
            s1 += logp[k, b]
        sf[t] = s1
        valid[t] = 1
    return sf, valid

# -------- per-strand regional aggregation --------
def aggregate_regions_on_sequence(seq: str, name: str, strand: str,
                                  logp: np.ndarray, fwd_table: np.ndarray,
                                  M: int, N: int,
                                  topA_heap: List[Tuple[float, tuple]],
                                  botB_heap: List[Tuple[float, tuple]],
                                  A: int, B: int):
    """
    Run scoring and regional aggregation on a single sequence (either + contig or its RC).
    For minus strand, call this with seq=RC(contig) and strand='-'; positions will be
    mapped back to original coordinates by the caller.
    """
    L = logp.shape[0]
    n = len(seq)
    if n < max(2*M+1, 2*N+1, L):
        return

    # k-mer scores (forward only, because seq is already orientation-specific)
    if NUMBA_OK:
        sf, valid = score_kmers_forward(base_to_code_array(seq), L, fwd_table)
    else:
        sf, valid = fallback_score_forward(seq, logp)

    gc_arr = gc_bool_array(seq)

    T = n - L + 1
    w_in  = max(0, (2*M + 1) - L + 1)
    w_out = max(0, (N - M) - L + 1)

    block = 5_000_000
    for i0 in range(0, n, block):
        i1 = min(n, i0 + block)
        idx = np.arange(i0, i1, dtype=np.int64)

        t_end_in = idx + (M - L + 1)
        t_end_ol = idx - (M + L)
        t_end_or = idx + (N - L + 1)

        # Inner windows
        t_min_in = max(0, int(t_end_in.min() - (w_in - 1)))
        t_max_in = min(T - 1, int(t_end_in.max()))
        if w_in > 0 and t_min_in <= t_max_in:
            sl_in = sf[t_min_in:t_max_in+1]
            vl_in = valid[t_min_in:t_max_in+1]
            ps_in = cumsum_f32_to_f64(sl_in) if NUMBA_OK else np.concatenate([[0.0], sl_in.cumsum(dtype=np.float64)])
            pc_in = cumsum_int8_to_int32(vl_in) if NUMBA_OK else np.concatenate([[0], vl_in.cumsum(dtype=np.int32)])
            te_in = (t_end_in - t_min_in).clip(min=0, max=t_max_in - t_min_in)
            sum_in = ps_in[te_in + 1] - ps_in[np.maximum(te_in - (w_in - 1), 0)]
            cnt_in = pc_in[te_in + 1] - pc_in[np.maximum(te_in - (w_in - 1), 0)]
        else:
            sum_in = np.zeros(idx.size, dtype=np.float64)
            cnt_in = np.zeros(idx.size, dtype=np.int32)

        # Outer windows (left)
        t_min_ol = max(0, int(t_end_ol.min() - (w_out - 1)))
        t_max_ol = min(T - 1, int(t_end_ol.max()))
        if w_out > 0 and t_min_ol <= t_max_ol:
            sl_ol = sf[t_min_ol:t_max_ol+1]
            vl_ol = valid[t_min_ol:t_max_ol+1]
            ps_ol = cumsum_f32_to_f64(sl_ol) if NUMBA_OK else np.concatenate([[0.0], sl_ol.cumsum(dtype=np.float64)])
            pc_ol = cumsum_int8_to_int32(vl_ol) if NUMBA_OK else np.concatenate([[0], vl_ol.cumsum(dtype=np.int32)])
            te_ol = (t_end_ol - t_min_ol).clip(min=0, max=t_max_ol - t_min_ol)
            sum_ol = ps_ol[te_ol + 1] - ps_ol[np.maximum(te_ol - (w_out - 1), 0)]
            cnt_ol = pc_ol[te_ol + 1] - pc_ol[np.maximum(te_ol - (w_out - 1), 0)]
        else:
            sum_ol = np.zeros(idx.size, dtype=np.float64)
            cnt_ol = np.zeros(idx.size, dtype=np.int32)

        # Outer windows (right)
        t_min_or = max(0, int(t_end_or.min() - (w_out - 1)))
        t_max_or = min(T - 1, int(t_end_or.max()))
        if w_out > 0 and t_min_or <= t_max_or:
            sl_or = sf[t_min_or:t_max_or+1]
            vl_or = valid[t_min_or:t_max_or+1]
            ps_or = cumsum_f32_to_f64(sl_or) if NUMBA_OK else np.concatenate([[0.0], sl_or.cumsum(dtype=np.float64)])
            pc_or = cumsum_int8_to_int32(vl_or) if NUMBA_OK else np.concatenate([[0], vl_or.cumsum(dtype=np.int32)])
            te_or = (t_end_or - t_min_or).clip(min=0, max=t_max_or - t_min_or)
            sum_or = ps_or[te_or + 1] - ps_or[np.maximum(te_or - (w_out - 1), 0)]
            cnt_or = pc_or[te_or + 1] - pc_or[np.maximum(te_or - (w_out - 1), 0)]
        else:
            sum_or = np.zeros(idx.size, dtype=np.float64)
            cnt_or = np.zeros(idx.size, dtype=np.int32)

        # Means
        with np.errstate(divide="ignore", invalid="ignore"):
            mean_in = np.where(cnt_in > 0, sum_in / cnt_in, np.nan)
            sum_o = sum_ol + sum_or
            cnt_o = cnt_ol + cnt_or
            mean_o = np.where(cnt_o > 0, sum_o / cnt_o, np.nan)

        # GC% from bases (identical whether + or − because it's the same bases)
        a0 = max(0, i0 - N)
        a1 = min(n, i1 + N)
        gc_slice = gc_arr[a0:a1]
        gc_ps = np.concatenate([[0], gc_slice.cumsum(dtype=np.int32)])

        L_in = (np.minimum(idx + M, n-1) - np.maximum(idx - M, 0) + 1).astype(np.int32)
        s_in = np.maximum(idx - M, 0) - a0
        e_in = np.minimum(idx + M, n-1) - a0
        gc_in = (gc_ps[e_in + 1] - gc_ps[s_in]).astype(np.int32)
        gc_in_pct = np.where(L_in > 0, gc_in * 100.0 / L_in, np.nan)

        s_ol = np.maximum(idx - N, 0) - a0
        e_ol = np.maximum(idx - M - 1, -1) - a0
        s_or = np.minimum(idx + M + 1, n) - a0
        e_or = np.minimum(idx + N, n-1) - a0
        e_ol_clip = np.maximum(e_ol, -1)
        len_ol = (e_ol_clip - s_ol + 1).clip(min=0)
        len_or = (e_or - s_or + 1).clip(min=0)
        gc_ol = np.where(len_ol > 0, gc_ps[e_ol_clip + 1] - gc_ps[s_ol], 0)
        gc_or = np.where(len_or > 0, gc_ps[e_or + 1] - gc_ps[s_or], 0)
        len_o = len_ol + len_or
        gc_o = gc_ol + gc_or
        gc_o_pct = np.where(len_o > 0, gc_o * 100.0 / len_o, np.nan)

        pos1 = idx + 1  # 1-based in THIS sequence (plus or RC)
        for j in range(idx.size):
            if cnt_in[j] <= 0:
                continue
            rec = (
                name,
                int(pos1[j]),    # 1-based in this orientation; caller remaps if strand == '-'
                strand,
                float(mean_in[j]),
                float(mean_o[j]) if not np.isnan(mean_o[j]) else float("nan"),
                float(gc_in_pct[j]), float(gc_o_pct[j]),
                int(cnt_in[j]), int(cnt_o[j])
            )
            score = float(mean_in[j])
            # Heaps keyed by score
            if len(topA_heap) < A:
                heapq.heappush(topA_heap, (score, rec))
            else:
                if score > topA_heap[0][0]:
                    heapq.heapreplace(topA_heap, (score, rec))
            neg = -score
            if len(botB_heap) < B:
                heapq.heappush(botB_heap, (neg, rec))
            else:
                if neg > botB_heap[0][0]:
                    heapq.heapreplace(botB_heap, (neg, rec))

# -------- output --------
def write_tsv_gz(path: str, rows: List[Tuple]):
    with gzip.open(path, "wt") as gz:
        gz.write(
            "chrom\tpos1\tstrand\tinner_mean_logPWM\touter_mean_logPWM\t"
            "GC_inner_pct\tGC_outer_pct\tkmer_count_inner\tkmer_count_outer\n"
        )
        for r in rows:
            gz.write("\t".join(map(str, r)) + "\n")

# -------- main --------
def main():
    ap = argparse.ArgumentParser(description="Strand-separated RC-exact genome regional PWM+GC scanner (LLR-ready).")
    ap.add_argument("--fasta", required=True, help="Genome FASTA (.fa or .fa.gz)")
    ap.add_argument("--pwm", required=True, help="PWM txt with columns Pos,A,C,G,U/T")
    ap.add_argument("-M", type=int, default=50, help="Inner half-width (default 50)")
    ap.add_argument("-N", type=int, default=500, help="Outer half-width (default 500; must be >M)")
    ap.add_argument("-A", type=int, default=2_000_000, help="Top-A strand entries to keep (default 2e6)")
    ap.add_argument("-B", type=int, default=2_000_000, help="Bottom-B strand entries to keep (default 2e6)")
    ap.add_argument("--bg", type=str, default=None, help='Background probs "A,C,G,T" for LLR')
    ap.add_argument("--bg-from-fasta", action="store_true", help="Estimate background q from FASTA (overrides --bg)")
    ap.add_argument("--out-prefix", required=True, help="Output prefix")
    args = ap.parse_args()

    if args.N <= args.M:
        ap.error("Require N > M so that outer flanks are non-empty.")

    logp = read_pwm_txt(args.pwm)
    bg_q = None
    if args.bg_from_fasta:
        bg_q = estimate_bg_from_fasta(args.fasta)
    elif args.bg is not None:
        bg_q = parse_bg_arg(args.bg)
    logp = apply_background_llr(logp, bg_q)

    # Precompute forward table once
    fwd_table = build_kmer_table_fwd(logp)

    topA_heap: List[Tuple[float, tuple]] = []
    botB_heap: List[Tuple[float, tuple]] = []

    for name, seq_plus in fasta_records(args.fasta):
        sys.stderr.write(f"[info] Processing {name} len={len(seq_plus):,} L={logp.shape[0]} M={args.M} N={args.N}\n")

        # (+) strand: scan the native contig
        aggregate_regions_on_sequence(seq_plus, name, '+', logp, fwd_table,
                                      args.M, args.N, topA_heap, botB_heap, args.A, args.B)

        # (−) strand: scan the reverse-complement as a separate sequence
        seq_minus = reverse_complement(seq_plus)
        aggregate_regions_on_sequence(seq_minus, name, '-', logp, fwd_table,
                                      args.M, args.N, topA_heap, botB_heap, args.A, args.B)

    # Materialize and map minus-strand coordinates back to original
    topA = sorted(topA_heap, key=lambda x: -x[0])
    botB = sorted(botB_heap, key=lambda x: x[0])  # stored as negative

    def remap_rows(rows: List[Tuple]) -> List[Tuple]:
        # rows contain (score, (chrom, pos1_in_orientation, strand, ...))
        # For '-' entries, convert pos1_in_orientation (on RC) back to original coordinate.
        out = []
        # To remap, we need contig lengths. Build one more pass to collect lengths.
        # (We could have cached during first pass; do a cheap re-scan.)
        contig_len = {}
        for cname, seq in fasta_records(args.fasta):
            contig_len[cname] = len(seq)
        for _, rec in rows:
            chrom, pos1o, strand, inner_mean, outer_mean, gc_in, gc_out, kc_in, kc_out = rec
            if strand == '-':
                n = contig_len[chrom]
                pos1 = n - pos1o + 1
            else:
                pos1 = pos1o
            out.append((chrom, pos1, strand, inner_mean, outer_mean, gc_in, gc_out, kc_in, kc_out))
        return out

    topA_rows = remap_rows(topA)
    botB_rows = remap_rows(botB)

    outA = f"{args.out_prefix}_topA.tsv.gz"
    outB = f"{args.out_prefix}_botB.tsv.gz"
    write_tsv_gz(outA, topA_rows)
    write_tsv_gz(outB, botB_rows)
    sys.stderr.write(f"[done] Wrote {outA} and {outB}\n")

if __name__ == "__main__":
    main()

