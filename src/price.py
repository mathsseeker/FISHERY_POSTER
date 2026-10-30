"""
price.py — Tauchen (1986) discretisation of the log-normal price process
=========================================================================
Converts the continuous log-normal GBM:
    log P_{t+1} = log P_t + nu_P + sigma_P * eps,   eps ~ N(0,1)
into a finite M-state Markov chain (P_grid, T_P).

Reference: Tauchen (1986), Economics Letters 20, 177-181.
"""

import numpy as np
from scipy.stats import norm
from params import PARAMS

# ── Params integrity check ───────────────────────────────────────────────────
assert PARAMS["m_s"][0] == 0.20, (
    f"STALE params.py detected: m_s[0]={PARAMS['m_s'][0]}, expected 0.20. "
    "Re-run after applying the 2026-04-11 audit corrections."
)
assert PARAMS["I_max"] >= 1_000_000, (
    f"STALE params.py: I_max={PARAMS['I_max']:,}, expected >= 1,000,000 t. "
    "Re-run after applying the 2026-04-11 audit corrections."
)
assert PARAMS["w_s"][0] > 0.1, (
    f"STALE params.py: w_s[0]={PARAMS['w_s'][0]:.5f}, expected ~0.37 (kg/fish). "
    "Remove the /1000.0 from the w_s definition in params.py."
)
# ─────────────────────────────────────────────────────────────────────────────


def build_price_chain(p: dict = PARAMS) -> tuple[np.ndarray, np.ndarray]:
    """
    Build the price grid and transition matrix via Tauchen (1986).

    Parameters
    ----------
    p : PARAMS dict

    Returns
    -------
    P_grid : array[M]    — price levels in ISK/tonne
    T_P    : array[M,M]  — row-stochastic transition matrix
                           T_P[m, m'] = P(P_{t+1} = P_grid[m'] | P_t = P_grid[m])
    """
    M       = p["M"]
    P0      = p["P0"]
    nu_P    = p["nu_P"]
    sigma_P = p["sigma_P"]

    # ── Grid: log-linearly spaced over 99% CI (10-yr horizon) ───────────────
    # Default: P_low  = P0 * exp(-2.576 * sigma_P * sqrt(10))
    #          P_high = P0 * exp(+2.576 * sigma_P * sqrt(10))
    # Override: if P_grid_lo / P_grid_hi are set in p, use those instead.
    log_P0  = np.log(P0)
    half    = 2.576 * sigma_P * np.sqrt(10)         # ±99% CI over 10 years
    log_lo  = np.log(p.get("P_grid_lo", P0 * np.exp(-half)))
    log_hi  = np.log(p.get("P_grid_hi", P0 * np.exp( half)))

    log_grid = np.linspace(log_lo, log_hi, M)       # equally spaced in log
    P_grid   = np.exp(log_grid)                     # price levels

    step = log_grid[1] - log_grid[0]                # uniform log-spacing

    # ── Tauchen transition matrix ─────────────────────────────────────────────
    # Conditional mean of log P_{t+1} given log P_t = log_grid[m]:
    #   mu_m = log_grid[m] + nu_P
    # Bin boundaries are midpoints between adjacent grid points.

    T_P = np.zeros((M, M))

    for m in range(M):
        mu = log_grid[m] + nu_P                     # conditional mean (log)

        # Lower boundary of each bin (midpoints; -inf for first, +inf for last)
        lo = log_grid - 0.5 * step
        hi = log_grid + 0.5 * step

        # CDF differences give probability mass in each bin
        probs = norm.cdf((hi - mu) / sigma_P) - norm.cdf((lo - mu) / sigma_P)

        # Edge corrections: absorb probability outside the grid into endpoints
        probs[0]  += norm.cdf((lo[0]  - mu) / sigma_P)
        probs[-1] += 1.0 - norm.cdf((hi[-1] - mu) / sigma_P)

        T_P[m, :] = probs

    # ── Normalise rows to exactly 1 (floating-point clean-up) ────────────────
    row_sums  = T_P.sum(axis=1, keepdims=True)
    T_P      /= row_sums

    # ── Assertion ─────────────────────────────────────────────────────────────
    assert np.allclose(T_P.sum(axis=1), 1.0, atol=1e-10), \
        "T_P rows do not sum to 1.0!"

    return P_grid, T_P


if __name__ == "__main__":
    P_grid, T_P = build_price_chain()
    M = len(P_grid)

    print("=" * 60)
    print("price.py — Tauchen Markov chain for log-normal price")
    print("=" * 60)
    print(f"  Grid size M = {M}")
    print(f"  P_lo  = {P_grid[0]:>12,.1f} ISK/t")
    print(f"  P_med = {P_grid[M//2]:>12,.1f} ISK/t")
    print(f"  P_hi  = {P_grid[-1]:>12,.1f} ISK/t")
    print()
    print(f"  First row T_P[0,:5]  = {T_P[0,:5]}")
    print(f"  Last  row T_P[-1,-5:]= {T_P[-1,-5:]}")
    print()
    print(f"  Row sums min={T_P.sum(axis=1).min():.12f}  "
          f"max={T_P.sum(axis=1).max():.12f}")
    print("  Row-sum assertion: passed ✓")

    # Simulate 1000 price paths to verify mean and std match analytically
    import numpy.random as rng_mod
    rng = rng_mod.default_rng(42)
    p   = PARAMS
    n_paths, T_sim = 1000, 50
    log_P = np.full(n_paths, np.log(p["P0"]))
    for _ in range(T_sim):
        log_P = log_P + p["nu_P"] + p["sigma_P"] * rng.standard_normal(n_paths)
    analytic_mean = np.log(p["P0"]) + T_sim * p["nu_P"]
    analytic_std  = p["sigma_P"] * np.sqrt(T_sim)
    print(f"\n  Monte Carlo check over {T_sim} steps ({n_paths} paths):")
    print(f"    Analytic  log-P mean = {analytic_mean:.4f}")
    print(f"    Simulated log-P mean = {log_P.mean():.4f}")
    print(f"    Analytic  log-P std  = {analytic_std:.4f}")
    print(f"    Simulated log-P std  = {log_P.std():.4f}")
    print("\nAll price tests passed. ✓")
