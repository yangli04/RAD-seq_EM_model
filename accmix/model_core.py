from __future__ import annotations
import argparse
import json
from typing import Optional, Tuple, Dict, List
import polars as pl
import numpy as np
from scipy import stats

# ---- Internalized pieces from scripts/model.py ----
def softclip(x: np.ndarray, lo: float = -40.0, hi: float = 40.0) -> np.ndarray:
    return np.clip(x, lo, hi)

def sigmoid(eta: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-softclip(eta)))

def add_intercept(X: np.ndarray) -> np.ndarray:
    return np.column_stack([np.ones((X.shape[0], 1)), X])

def irls_logistic(X: np.ndarray, r: np.ndarray, max_iter: int = 100, tol: float = 1e-6) -> np.ndarray:
    n, p = X.shape
    beta = np.zeros(p)
    for _ in range(max_iter):
        eta = X @ beta
        p_hat = sigmoid(eta)
        w = np.clip(p_hat * (1 - p_hat), 1e-9, None)
        z = eta + (r - p_hat) / np.clip(w, 1e-12, None)
        WX = X * w[:, None]
        H = X.T @ WX
        g = X.T @ (w * z)
        try:
            beta_new = np.linalg.solve(H, g)
        except np.linalg.LinAlgError:
            beta_new = np.linalg.solve(H + 1e-6 * np.eye(p), g)
        if np.linalg.norm(beta_new - beta) < tol * (1.0 + np.linalg.norm(beta)):
            beta = beta_new
            break
        beta = beta_new
    return beta

def read_polars_input(df: pl.DataFrame, site_col: str, s_col: str, feat_cols: List[str]) -> Tuple[pl.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    required = {site_col, s_col, *feat_cols}
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    required_cols = [site_col, s_col] + feat_cols
    df = df.drop_nulls(subset=required_cols)
    s = df[s_col].to_numpy()
    finite_mask = np.isfinite(s)
    if not np.all(finite_mask):
        df = df.filter(pl.Series("finite_mask", finite_mask))
        s = s[finite_mask]
    X = df.select(feat_cols).to_numpy()
    X = add_intercept(X)
    return df, s, X, df[site_col].to_numpy()

def run_em_Gaussian(df: pl.DataFrame, s: np.ndarray, X: np.ndarray,
                    max_iter: int = 900, tol: float = 1e-5) -> Tuple[Dict, np.ndarray, np.ndarray]:
    n = len(s)
    q25, q50, q75 = np.percentile(s, [25, 50, 75])
    mask_bg = s <= q50
    mask_sig = s > q50
    mu0 = np.mean(s[mask_bg]) if np.any(mask_bg) else q25
    sigma0 = np.std(s[mask_bg]) if np.any(mask_bg) else (q50 - q25) / 2
    mu1 = np.mean(s[mask_sig]) if np.any(mask_sig) else q75
    sigma1 = np.std(s[mask_sig]) if np.any(mask_sig) else (q75 - q50) / 2
    sigma0 = max(sigma0, 0.1)
    sigma1 = max(sigma1, 0.1)

    r = np.zeros(n)
    for i in range(n):
        p_bg = stats.norm.pdf(s[i], mu0, sigma0)
        p_sig = stats.norm.pdf(s[i], mu1, sigma1)
        r[i] = p_sig / (p_bg + p_sig + 1e-10)
    r = np.clip(r, 0.01, 0.99)
    beta = irls_logistic(X, r, max_iter=20)

    log_likelihood_prev = -np.inf
    for iteration in range(max_iter):
        p = sigmoid(X @ beta)
        eps = 1e-10
        sigma0 = max(sigma0, eps)
        sigma1 = max(sigma1, eps)
        log_p_bg = stats.norm.logpdf(s, mu0, sigma0)
        log_p_sig = stats.norm.logpdf(s, mu1, sigma1)
        p_bg = np.exp(log_p_bg)
        p_sig = np.exp(log_p_sig)
        numerator = p * p_sig
        denominator = (1 - p) * p_bg + p * p_sig + eps
        r = numerator / denominator
        r = np.clip(r, eps, 1 - eps)
        try:
            beta = irls_logistic(X, r, max_iter=20)
        except:
            beta = beta * 0.95
        w0 = 1 - r
        sum_w0 = np.sum(w0) + eps
        mu0 = np.sum(w0 * s) / sum_w0
        sigma0_sq = np.sum(w0 * (s - mu0)**2) / sum_w0
        sigma0 = np.sqrt(max(sigma0_sq, eps))
        w1 = r
        sum_w1 = np.sum(w1) + eps
        mu1 = np.sum(w1 * s) / sum_w1
        sigma1_sq = np.sum(w1 * (s - mu1)**2) / sum_w1
        sigma1 = np.sqrt(max(sigma1_sq, eps))
        p_updated = sigmoid(X @ beta)
        log_p_bg_updated = stats.norm.logpdf(s, mu0, sigma0)
        log_p_sig_updated = stats.norm.logpdf(s, mu1, sigma1)
        max_log = np.maximum(
            np.log(1 - p_updated + eps) + log_p_bg_updated,
            np.log(p_updated + eps) + log_p_sig_updated
        )
        log_likelihood = np.sum(
            max_log + np.log(
                np.exp(np.log(1 - p_updated + eps) + log_p_bg_updated - max_log) +
                np.exp(np.log(p_updated + eps) + log_p_sig_updated - max_log)
            )
        )
        if abs(log_likelihood - log_likelihood_prev) < tol * (1 + abs(log_likelihood_prev)):
            break
        log_likelihood_prev = log_likelihood

    p_final = sigmoid(X @ beta)
    p_bg_final = stats.norm.pdf(s, mu0, sigma0)
    p_sig_final = stats.norm.pdf(s, mu1, sigma1)
    denominator_final = (1 - p_final) * p_bg_final + p_final * p_sig_final + 1e-10
    r_final = (p_final * p_sig_final) / denominator_final
    result = {
        "mode": "em_gaussian",
        "beta": beta.tolist(),
        "gaussian_params": {
            "mu0": float(mu0), "sigma0": float(sigma0),
            "mu1": float(mu1), "sigma1": float(sigma1)
        },
        "diagnostics": {
            "iters": iteration + 1,
            "converged": iteration < max_iter - 1,
            "final_log_likelihood": float(log_likelihood),
            "mean_p": float(p_final.mean()),
            "mean_r": float(r_final.mean())
        }
    }
    return result, p_final, r_final

def fit_model(input_parquet: str,
              out_prefix: Optional[str] = None,
              rbp_name: str = "RBP",
              motif_id: str = "Motif",
              feature_columns: Optional[List[str]] = None,
              site_col: str = "id",
              s_col: str = "s_l"):
    feature_columns = feature_columns or ['inner_mean_logPWM', 'outer_mean_logPWM', 'GC_inner_pct', 'GC_outer_pct',
                                          'TSS_proximity', 'PhastCons100_percent', 'TPM', 'score_phastcons100', 'score_phylop100']
    skip_normalization = ['TSS_proximity', 'PhastCons100_percent', 'score_phastcons100']
    df = pl.read_parquet(input_parquet)
    for c in [site_col, s_col] + feature_columns:
        if c not in df.columns:
            raise SystemExit(f"Missing column {c} in {input_parquet}")
    df_pl, s, X, _ = read_polars_input(df, site_col, s_col, feature_columns)
    X_z = X.copy()
    for i, col in enumerate(feature_columns):
        if col not in skip_normalization:
            col_z = stats.zscore(X[:, i])
            X_z[:, i] = np.nan_to_num(col_z, nan=0.0)
    X_z[:, 0] = 1
    log_s = np.log(s + 1)
    model, p, r = run_em_Gaussian(df_pl, log_s, X_z)
    out_df = df_pl.with_columns([
        pl.Series("prior_p", p),
        pl.Series("posterior_r", r),
    ])
    prefix = out_prefix or f"results/{rbp_name}_{motif_id}"
    out_parquet = f"{prefix}.model.parquet"
    out_df.write_parquet(out_parquet)
    with open(f"{prefix}.model.json", "w") as fh:
        json.dump(model, fh, indent=2)
    return out_parquet, f"{prefix}.model.json"


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="accmix model")
    ap.add_argument("--input-parquet", required=True)
    ap.add_argument("--out-prefix", default=None)
    ap.add_argument("--rbp-name", default="RBP")
    ap.add_argument("--motif-id", default="Motif")
    return ap
