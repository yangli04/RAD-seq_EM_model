#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Center–Flank Depth-Weighted Mixture with Logistic Prior

Modes:
  - empirical : Γ0, Γ1 from subsets (MoM), then optimize β by empirical marginal (no EM)
  - em        : classic EM; IRLS for β, MoM for Γz (fast, approximate M-step for Γ)
  - mle       : joint MLE for β, Γ0, Γ1 via L-BFGS-B (observed-data likelihood)
  - hybrid    : FIX Γ0 from BG subset (MoM), then MLE for β and Γ1 with Γ0 fixed

Input TSV must have:
  - a unique site id column (default: 'site')
  - s_l column (default: 's')  [already computed = t_N^2 + t_F^2, per your note]
  - prior feature columns (Gl) passed via --feat-cols (comma-separated)

Outputs:
  - <out>.parquet : input + columns [prior_p, posterior_r]
  - <out>.json    : fitted parameters (beta, gamma params), diagnostics
"""
from __future__ import annotations
import argparse, json, math, sys
from typing import Dict, Tuple, List, Optional

import numpy as np
import polars as pl
from scipy.optimize import minimize
from scipy import stats
from scipy.special import digamma as psi


def softclip(x: np.ndarray, lo: float = -40.0, hi: float = 40.0) -> np.ndarray:
    return np.clip(x, lo, hi)

def sigmoid(eta: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-softclip(eta)))

def logit(p: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    p = np.clip(p, eps, 1 - eps)
    return np.log(p) - np.log1p(-p)

def add_intercept(X: np.ndarray) -> np.ndarray:
    return np.column_stack([np.ones((X.shape[0], 1)), X])

def gammalogpdf(x: np.ndarray, k: float, theta: float, eps: float = 1e-12) -> np.ndarray:
    x = np.clip(x, eps, None)
    return (k - 1.0) * np.log(x) - (x / theta) - math.lgamma(k) - k * np.log(theta)

def _gamma_logpdf_and_sufficient(s: np.ndarray, k: float, th: float):
    logf = (k - 1.0) * np.log(s) - (s / th) - math.lgamma(k) - k * np.log(th)
    dL_dk_term  = np.log(s) - psi(k) - np.log(th)
    dL_dth_term = (s - k * th) / (th * th)
    return logf, dL_dk_term, dL_dth_term

def mom_gamma(x: np.ndarray, w: Optional[np.ndarray] = None, eps: float = 1e-12) -> Tuple[float, float]:
    """Method-of-moments Gamma fit (k, theta) with optional weights."""
    if w is None:
        w = np.ones_like(x)
    w = w.astype(float)
    x = x.astype(float)
    W = w.sum()
    m = (w * x).sum() / max(W, eps)
    v = (w * (x - m) ** 2).sum() / max(W, eps)
    v = max(v, eps)
    theta = v / max(m, eps)
    k = (m ** 2) / v
    k = max(k, 1e-6)
    theta = max(theta, 1e-12)
    return float(k), float(theta)


def irls_logistic(X: np.ndarray, r: np.ndarray, max_iter: int = 100, tol: float = 1e-6) -> np.ndarray:
    """
    Weighted logistic regression for M-step in EM:
      maximize sum_l r_l log p_l + (1-r_l) log(1-p_l)
    """
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

def grad_empirical_beta(X: np.ndarray, Bm1: np.ndarray, beta: np.ndarray) -> Tuple[float, np.ndarray]:
    """
    Empirical objective:
      L(beta) = sum log(1 + p(B-1)),  p = sigmoid(X beta)
    Returns (L, grad)
    """
    eta = X @ beta
    p = sigmoid(eta)
    denom = 1.0 + p * Bm1
    L = np.sum(np.log(denom))
    alpha = (Bm1 * p * (1 - p)) / denom
    grad = X.T @ alpha
    return float(L), grad

def maximize_empirical_beta(X: np.ndarray, Bm1: np.ndarray,
                            max_iter: int = 500, tol: float = 1e-6,
                            step0: float = 1.0) -> np.ndarray:
    """Backtracking gradient ascent for empirical β."""
    beta = np.zeros(X.shape[1])
    L_prev = -np.inf
    for _ in range(max_iter):
        L, g = grad_empirical_beta(X, Bm1, beta)
        if L - L_prev < tol * (1.0 + abs(L_prev)):
            break
        step = step0
        while True:
            beta_try = beta + step * g
            L_try, _ = grad_empirical_beta(X, Bm1, beta_try)
            if L_try >= L + 1e-4 * step * np.dot(g, g) or step < 1e-8:
                beta, L_prev = beta_try, L_try
                break
            step *= 0.5
    return beta

# --------------------------- IO & subset ---------------------------

def read_polars_input(df: pl.DataFrame, site_col: str, s_col: str, feat_cols: List[str]) -> Tuple[pl.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    required = {site_col, s_col, *feat_cols}
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    
    required_cols = [site_col, s_col] + feat_cols
    original_len = df.height
    df = df.drop_nulls(subset=required_cols)
    filtered_len = df.height
    
    if filtered_len < original_len:
        print(f"Dropped {original_len - filtered_len} rows with null values ({filtered_len} remaining)")
    
    s = df[s_col].to_numpy()

    finite_mask = np.isfinite(s)
    if not np.all(finite_mask):
        print(f"Warning: {np.sum(~finite_mask)} non-finite values in {s_col} column")
        s = s[finite_mask]
        # Also filter other arrays accordingly
        df = df.filter(pl.Series("finite_mask", finite_mask))
    
    print(f"Final data shape: {df.height} rows")
    print(f"s_col ({s_col}) stats: min={np.min(s):.2e}, max={np.max(s):.2e}, median={np.nanmedian(s):.2e}")
    # G columns == feat_cols.
    X = df.select(feat_cols).to_numpy()
    X = add_intercept(X)
    return df, s, X, df[site_col].to_numpy()

def choose_bg_alt_indices(df: pl.DataFrame,
                          pwm_col: Optional[str],
                          bg_quantile: float,
                          alt_quantile: float,
                          limit_top_pool: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Heuristic subset chooser:
    - If pwm_col is present: BG = low PWM + lower tail within top pool; ALT = high PWM.
    - Else: BG = low s tail; ALT = high s tail.
    """
    n = df.height
    if pwm_col is None or pwm_col not in df.columns:
        s = df["s"].to_numpy()
        q_bg = np.quantile(s, bg_quantile)
        q_alt = np.quantile(s, alt_quantile)
        bg = np.where(s <= q_bg)[0]
        alt = np.where(s >= q_alt)[0]
        return bg, alt

    pwm = df[pwm_col].to_numpy()
    idx_sorted = np.argsort(pwm)
    k_low = max(1, int(bg_quantile * n))
    low_pool = idx_sorted[:k_low]

    high_pool = idx_sorted[::-1]
    if limit_top_pool is not None:
        high_pool = high_pool[:min(limit_top_pool, n)]
    m_bg_tail = max(1, int(0.2 * len(high_pool)))
    add_bg = high_pool[-m_bg_tail:]

    bg = np.unique(np.concatenate([low_pool, add_bg]))
    m_alt = max(1, int((1 - alt_quantile) * n))
    alt = idx_sorted[-m_alt:]
    return bg, alt


def run_em(df: pl.DataFrame, s: np.ndarray, X: np.ndarray,
           max_iter: int = 900, tol: float = 1e-5) -> Tuple[Dict, np.ndarray, np.ndarray]:
    # init via s median - use nanmedian to handle NaN values
    med = float(np.nanmedian(s))
    if np.isnan(med):
        # If all values are NaN, use a fallback
        med = 1.0
    r = (s > med).astype(float) * 0.9 + 0.05
    k0, t0 = mom_gamma(s, 1 - r)
    k1, t1 = mom_gamma(s, r)
    beta = irls_logistic(X, r)

    L_prev = -np.inf
    for it in range(max_iter):
        p = sigmoid(X @ beta)
        f0 = np.exp(gammalogpdf(s, k0, t0))
        f1 = np.exp(gammalogpdf(s, k1, t1))
        numer = p * f1
        denom = numer + (1 - p) * f0 + 1e-300
        r = numer / denom

        beta = irls_logistic(X, r)
        k0, t0 = mom_gamma(s, 1 - r)
        k1, t1 = mom_gamma(s, r)

        L = float(np.sum(np.log(denom)))
        if L - L_prev < tol * (1.0 + abs(L_prev)):
            break
        L_prev = L

    p = sigmoid(X @ beta)
    f0 = np.exp(gammalogpdf(s, k0, t0))
    f1 = np.exp(gammalogpdf(s, k1, t1))
    denom = (1 - p) * f0 + p * f1 + 1e-300
    r = (p * f1) / denom

    out = dict(
        mode="em",
        beta=beta.tolist(),
        gamma_params={"k0": float(k0), "theta0": float(t0), "k1": float(k1), "theta1": float(t1)},
        diagnostics={"iters": it + 1, "mean_p": float(p.mean()), "mean_r": float(r.mean())}
    )
    return out, p, r

def run_joint_mle(df: pl.DataFrame, s: np.ndarray, X: np.ndarray,
                  init: Optional[Dict] = None,
                  max_iter: int = 500, tol: float = 1e-6) -> Tuple[Dict, np.ndarray, np.ndarray]:
    """Joint MLE for β, (k0,θ0), (k1,θ1)."""
    n, p = X.shape
    s_clip = np.clip(s, 1e-300, None)

    if init is None:
        em_model, _, _ = run_em(df, s, X, max_iter=50, tol=1e-4)
        beta0 = np.asarray(em_model["beta"])
        k0 = em_model["gamma_params"]["k0"]; t0 = em_model["gamma_params"]["theta0"]
        k1 = em_model["gamma_params"]["k1"]; t1 = em_model["gamma_params"]["theta1"]
    else:
        beta0 = np.asarray(init["beta"])
        k0, t0 = init["k0"], init["theta0"]
        k1, t1 = init["k1"], init["theta1"]

    def pack(beta, k0, t0, k1, t1):
        return np.concatenate([beta, np.array([k0, t0, k1, t1], float)])

    def unpack(w):
        beta = w[:p]; k0, t0, k1, t1 = w[p:]
        return beta, float(k0), float(t0), float(k1), float(t1)

    w0 = pack(beta0, k0, t0, k1, t1)
    bounds = [(None, None)] * p + [(1e-6, None), (1e-9, None), (1e-6, None), (1e-9, None)]

    def objective_and_grad(w):
        beta, k0, t0, k1, t1 = unpack(w)
        p_i = sigmoid(X @ beta)
        logf0, d0dk, d0dt = _gamma_logpdf_and_sufficient(s_clip, k0, t0)
        logf1, d1dk, d1dt = _gamma_logpdf_and_sufficient(s_clip, k1, t1)
        f0, f1 = np.exp(logf0), np.exp(logf1)

        denom = (1 - p_i) * f0 + p_i * f1 + 1e-300
        loglik = np.sum(np.log(denom))
        r1 = (p_i * f1) / denom
        dl_deta = ((f1 - f0) * p_i * (1 - p_i)) / denom
        g_beta = X.T @ dl_deta
        g_k0, g_t0 = np.sum((1 - r1) * d0dk), np.sum((1 - r1) * d0dt)
        g_k1, g_t1 = np.sum(r1 * d1dk), np.sum(r1 * d1dt)
        grad = np.concatenate([g_beta, np.array([g_k0, g_t0, g_k1, g_t1], float)])
        return -loglik, -grad

    res = minimize(lambda w: objective_and_grad(w)[0], w0,
                   method="L-BFGS-B", jac=lambda w: objective_and_grad(w)[1],
                   bounds=bounds, options=dict(maxiter=max_iter, ftol=tol))

    beta_ml, k0_ml, t0_ml, k1_ml, t1_ml = unpack(res.x)
    p_ml = sigmoid(X @ beta_ml)
    f0 = np.exp(gammalogpdf(s_clip, k0_ml, t0_ml))
    f1 = np.exp(gammalogpdf(s_clip, k1_ml, t1_ml))
    denom = (1 - p_ml) * f0 + p_ml * f1 + 1e-300
    r_ml = (p_ml * f1) / denom

    out = dict(
        mode="mle",
        success=bool(res.success),
        message=str(res.message),
        beta=beta_ml.tolist(),
        gamma_params={"k0": float(k0_ml), "theta0": float(t0_ml), "k1": float(k1_ml), "theta1": float(t1_ml)},
        diagnostics={"iters": int(res.nit), "mean_p": float(p_ml.mean()), "mean_r": float(r_ml.mean())}
    )
    return out, p_ml, r_ml

# Add this function after the existing run_em function

def normallogpdf(x: np.ndarray, mu: float, sigma: float, eps: float = 1e-12) -> np.ndarray:
    """Log PDF of normal distribution."""
    sigma = max(sigma, eps)
    return -0.5 * np.log(2 * np.pi * sigma**2) - 0.5 * ((x - mu) / sigma)**2

def mom_normal(x: np.ndarray, w: Optional[np.ndarray] = None, eps: float = 1e-12) -> Tuple[float, float]:
    """Method-of-moments Normal fit (mu, sigma) with optional weights."""
    if w is None:
        w = np.ones_like(x)
    w = w.astype(float)
    x = x.astype(float)
    W = w.sum()
    mu = (w * x).sum() / max(W, eps)
    variance = (w * (x - mu) ** 2).sum() / max(W, eps)
    sigma = np.sqrt(max(variance, eps))
    return float(mu), float(sigma)

def run_em_Gaussian(df: pl.DataFrame, s: np.ndarray, X: np.ndarray,
                   max_iter: int = 900, tol: float = 1e-5, inverse: bool = False) -> Tuple[Dict, np.ndarray, np.ndarray]:
    """
    Improved EM algorithm using Normal (Gaussian) mixture model.
    
    Args:
        df: Input dataframe
        s: Observed values (should be log-transformed: log(s_l + 1))
        X: Feature matrix (with intercept column as all 1s)
        max_iter: Maximum EM iterations
        tol: Convergence tolerance
        inverse: If True, switches initialization so background gets higher values and signal gets lower values
    
    Returns:
        Tuple of (model_dict, prior_probabilities, posterior_probabilities)
    """
    n = len(s)
    p_features = X.shape[1]
    
    # Better initialization using quantiles
    q25, q50, q75 = np.percentile(s, [25, 50, 75])
    
    # Initialize mixture components based on data distribution
    if inverse:
        # When inverse=True, switch the masks: bg gets higher values, sig gets lower values
        mask_bg = s > q50
        mask_sig = s <= q50
        mu0 = np.mean(s[mask_bg]) if np.any(mask_bg) else q75
        sigma0 = np.std(s[mask_bg]) if np.any(mask_bg) else (q75 - q50) / 2
        mu1 = np.mean(s[mask_sig]) if np.any(mask_sig) else q25
        sigma1 = np.std(s[mask_sig]) if np.any(mask_sig) else (q50 - q25) / 2
    else:
        # Default: Background component (lower values), Signal component (higher values)
        mask_bg = s <= q50
        mask_sig = s > q50
        mu0 = np.mean(s[mask_bg]) if np.any(mask_bg) else q25
        sigma0 = np.std(s[mask_bg]) if np.any(mask_bg) else (q50 - q25) / 2
        mu1 = np.mean(s[mask_sig]) if np.any(mask_sig) else q75
        sigma1 = np.std(s[mask_sig]) if np.any(mask_sig) else (q75 - q50) / 2
    
    sigma0 = max(sigma0, 0.1)  # Minimum variance
    sigma1 = max(sigma1, 0.1)  # Minimum variance
    
    # Initialize responsibilities using soft assignment
    r = np.zeros(n)
    for i in range(n):
        # Probability of belonging to signal component
        p_bg = stats.norm.pdf(s[i], mu0, sigma0)
        p_sig = stats.norm.pdf(s[i], mu1, sigma1)
        r[i] = p_sig / (p_bg + p_sig + 1e-10)
    
    # Smooth initialization
    r = np.clip(r, 0.01, 0.99)
    
    # Initialize logistic regression coefficients
    beta = irls_logistic(X, r, max_iter=20)
    
    # EM iterations
    log_likelihood_prev = -np.inf
    
    for iteration in range(max_iter):
        # E-step: Update responsibilities
        # Prior probabilities from logistic regression
        linear_pred = X @ beta
        linear_pred = np.clip(linear_pred, -500, 500)
        p = sigmoid(linear_pred)
        
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
        
        # M-step: Update parameters
        
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
        
        if np.isnan(mu0) or np.isnan(mu1) or np.isnan(sigma0) or np.isnan(sigma1):
            print(f"Warning: NaN parameters at iteration {iteration}")
            break
    
    # Final E-step for output
    print(f"EM converged in {iteration+1} iterations with log-likelihood: {log_likelihood:.6f}")
    p_final = sigmoid(X @ beta)
    p_bg_final = stats.norm.pdf(s, mu0, sigma0)
    p_sig_final = stats.norm.pdf(s, mu1, sigma1)
    
    denominator_final = (1 - p_final) * p_bg_final + p_final * p_sig_final + 1e-10
    r_final = (p_final * p_sig_final) / denominator_final
    
    # Package results
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
            "mean_r": float(r_final.mean()),
            "separation": float(abs(mu1 - mu0) / (sigma0 + sigma1))  # Measure of component separation
        }
    }
    
    return result, p_final, r_final

def run_hybrid_mle_bgfixed(df: pl.DataFrame, s: np.ndarray, X: np.ndarray,
                           bg_idx: np.ndarray,
                           init_from_em: bool = True,
                           max_iter: int = 400, tol: float = 1e-6) -> Tuple[Dict, np.ndarray, np.ndarray]:
    """
    Your requested hybrid:
      1) Γ0 from BG subset (MoM),
      2) with Γ0 fixed, MLE for β and Γ1 via L-BFGS-B.
    """
    n, p = X.shape
    s_clip = np.clip(s, 1e-300, None)

    # Step1
    k0, t0 = mom_gamma(s[bg_idx])

    # Step2
    if init_from_em:
        em_model, _, _ = run_em(df, s, X, max_iter=30, tol=1e-4)
        beta0 = np.asarray(em_model["beta"])
        k1_0  = float(em_model["gamma_params"]["k1"])
        t1_0  = float(em_model["gamma_params"]["theta1"])
    else:
        beta0 = np.zeros(p)
        k1_0, t1_0 = mom_gamma(s)

    def pack(beta, k1, t1): return np.concatenate([beta, np.array([k1, t1], float)])
    def unpack(w): beta = w[:p]; k1, t1 = w[p:]; return beta, float(k1), float(t1)

    w0 = pack(beta0, k1_0, t1_0)
    bounds = [(None, None)] * p + [(1e-6, None), (1e-9, None)]

    def objective_and_grad(w):
        beta, k1, t1 = unpack(w)
        p_i = sigmoid(X @ beta)
        logf0, _, _ = _gamma_logpdf_and_sufficient(s_clip, k0, t0)    # fixed
        logf1, d1dk, d1dt = _gamma_logpdf_and_sufficient(s_clip, k1, t1)
        f0, f1 = np.exp(logf0), np.exp(logf1)

        denom = (1 - p_i) * f0 + p_i * f1 + 1e-300
        loglik = np.sum(np.log(denom))
        r1 = (p_i * f1) / denom
        dl_deta = ((f1 - f0) * p_i * (1 - p_i)) / denom
        g_beta = X.T @ dl_deta
        g_k1 = np.sum(r1 * d1dk)
        g_t1 = np.sum(r1 * d1dt)
        grad = np.concatenate([g_beta, np.array([g_k1, g_t1], float)])
        return -loglik, -grad

    res = minimize(lambda w: objective_and_grad(w)[0], w0,
                   method="L-BFGS-B", jac=lambda w: objective_and_grad(w)[1],
                   bounds=bounds, options=dict(maxiter=max_iter, ftol=tol))

    beta_ml, k1_ml, t1_ml = unpack(res.x)
    p_ml = sigmoid(X @ beta_ml)
    f0 = np.exp(gammalogpdf(s_clip, k0, t0))
    f1 = np.exp(gammalogpdf(s_clip, k1_ml, t1_ml))
    denom = (1 - p_ml) * f0 + p_ml * f1 + 1e-300
    r_ml = (p_ml * f1) / denom

    out = dict(
        mode="hybrid",
        success=bool(res.success),
        message=str(res.message),
        beta=beta_ml.tolist(),
        gamma_params={"k0": float(k0), "theta0": float(t0), "k1": float(k1_ml), "theta1": float(t1_ml)},
        diagnostics={"iters": int(res.nit), "mean_p": float(p_ml.mean()), "mean_r": float(r_ml.mean())}
    )
    return out, p_ml, r_ml

def main():
    ap = argparse.ArgumentParser(description="Center–Flank depth-weighted mixture with logistic prior.")
    ap.add_argument("--tsv", required=True, help="Input TSV, one row per site.")
    ap.add_argument("--site-col", default="site", help="Unique site id column.")
    ap.add_argument("--s-col", default="s", help="Column with s_l (precomputed).")
    ap.add_argument("--feat-cols", required=True, help="Comma-separated prior feature columns (G_l).")
    ap.add_argument("--mode", choices=["empirical", "em", "mle", "hybrid"], default="hybrid",
                    help="empirical | em | mle | hybrid (fix Γ0 from BG, MLE β & Γ1)")
    # subset chooser
    ap.add_argument("--pwm-col", default=None, help="PWM column for BG/ALT selection (optional).")
    ap.add_argument("--bg-quantile", type=float, default=0.10, help="Lower quantile for BG if no PWM (or first stage).")
    ap.add_argument("--alt-quantile", type=float, default=0.90, help="Upper quantile for ALT if used.")
    ap.add_argument("--limit-top-pool", type=int, default=1_000_000, help="Limit for high-PWM pool size (None disables).")
    # EM knobs
    ap.add_argument("--em-iters", type=int, default=100)
    ap.add_argument("--em-tol", type=float, default=1e-5)
    # Output
    ap.add_argument("--out-prefix", required=True, help="Output prefix for parquet/json.")
    args = ap.parse_args()

    feat_cols = [c.strip() for c in args.feat_cols.split(",") if c.strip()]
    df, s, X, site_ids = read_tsv_polars(args.tsv, args.site_col, args.s_col, feat_cols)


    if args.mode == "em":
        model, p, r = run_em(df, s, X, max_iter=args.em_iters, tol=args.em_tol)

    elif args.mode == "mle":
        model, p, r = run_joint_mle(df, s, X, init=None, max_iter=500, tol=1e-6)

    else:  # hybrid (default)
        # bg_idx: background indexes chose for estimating the background gamma (Z = 0).
        bg_idx, _ = choose_bg_alt_indices(
            df, args.pwm_col, args.bg_quantile, args.alt_quantile,
            args.limit_top_pool if args.limit_top_pool > 0 else None
        )
        model, p, r = run_hybrid_mle_bgfixed(df, s, X, bg_idx,
                                             init_from_em=True, max_iter=400, tol=1e-6)

    out_df = df.with_columns([
        pl.Series("prior_p", p),
        pl.Series("posterior_r", r),
    ])
    out_parquet = f"{args.out_prefix}.parquet"
    out_df.write_parquet(out_parquet)

    out_json = f"{args.out_prefix}.json"
    with open(out_json, "w") as fh:
        json.dump(model, fh, indent=2)

    print(f"Mode: {model['mode']}")
    if "diagnostics" in model and "iters" in model["diagnostics"]:
        print(f"Iters: {model['diagnostics']['iters']}")
    print(f"mean prior p: {model['diagnostics']['mean_p']:.4f} | mean posterior r: {model['diagnostics']['mean_r']:.4f}")
    print(f"Wrote: {out_parquet}")
    print(f"Wrote: {out_json}")

if __name__ == "__main__":
    main()
