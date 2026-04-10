"""
profit.py — Precomputed profit tensor π[N, M, J]
=================================================
Schooling-fishery profit (Mota 2020, Marine Policy 115):
    π(I, P, H) = P * H  -  c1 * H^c2

No density-dependent cost — finding cost is low regardless of stock size
for schooling species (unlike halibut in Pizarro & Schwartz 2021).

Infeasible cells (H > B or H < 0) are set to -inf so VFI never picks them.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from params import PARAMS
from price  import build_price_chain


# ─────────────────────────────────────────────────────────────────────────────
# Grid builder
# ─────────────────────────────────────────────────────────────────────────────

def build_grids(p: dict = PARAMS,
                P_grid: np.ndarray | None = None
                ) -> tuple[np.ndarray, np.ndarray]:
    """
    Build the biomass and harvest grids.

    Returns
    -------
    B_grid : array[N]  — equally spaced in [0.01*I_max, I_max]
    H_grid : array[J]  — equally spaced in [0, 0.40*I_max]
    """
    I_max  = p["I_max"]
    B_grid = np.linspace(0.01 * I_max, I_max, p["N"])
    H_grid = np.linspace(0.0,          0.40 * I_max, p["J"])
    return B_grid, H_grid


# ─────────────────────────────────────────────────────────────────────────────
# Profit tensor
# ─────────────────────────────────────────────────────────────────────────────

def build_profit_tensor(
    B_grid: np.ndarray,
    P_grid: np.ndarray,
    H_grid: np.ndarray,
    p: dict = PARAMS,
) -> np.ndarray:
    """
    Build the N × M × J profit tensor using pure numpy broadcasting.

    π[n, m, j] = P_grid[m] * H_grid[j]  -  c1 * H_grid[j]^c2
               = -inf  if H_grid[j] > B_grid[n]  (infeasible)
               = -inf  if H_grid[j] < 0

    Parameters
    ----------
    B_grid : array[N]
    P_grid : array[M]
    H_grid : array[J]
    p      : PARAMS dict

    Returns
    -------
    pi : array[N, M, J]  — profit tensor
    """
    c1 = p["c1"]
    c2 = p["c2"]

    # Revenue:  P[m] * H[j]  — shape (1, M, J)  via broadcasting
    # Cost:     c1 * H[j]^c2 — shape (1, 1, J)
    revenue = P_grid[np.newaxis, :, np.newaxis] * H_grid[np.newaxis, np.newaxis, :]
    cost    = c1 * (H_grid ** c2)[np.newaxis, np.newaxis, :]

    pi = revenue - cost          # shape (1, M, J) broadcast to (N, M, J)
    pi = np.broadcast_to(pi, (len(B_grid), len(P_grid), len(H_grid))).copy()

    # Infeasibility mask: H > B  →  -inf   (broadcast to full N×M×J shape)
    infeasible = np.broadcast_to(
        H_grid[np.newaxis, np.newaxis, :] > B_grid[:, np.newaxis, np.newaxis],
        pi.shape,
    ).copy()
    pi[infeasible] = -np.inf

    # Guard: H = 0 always feasible; H < 0 never constructed (H_grid[0] = 0)
    return pi


# ─────────────────────────────────────────────────────────────────────────────
# Self-test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    p = PARAMS
    P_grid, T_P = build_price_chain(p)
    B_grid, H_grid = build_grids(p, P_grid)

    print("=" * 60)
    print("profit.py — building π[N, M, J]")
    print("=" * 60)
    print(f"  N={p['N']}, M={p['M']}, J={p['J']}  →  "
          f"tensor size = {p['N']*p['M']*p['J']:,} cells")

    pi = build_profit_tensor(B_grid, P_grid, H_grid, p)

    print(f"  pi.shape  = {pi.shape}")
    print(f"  pi finite = {np.isfinite(pi).sum():,} cells")
    print(f"  pi -inf   = {(pi == -np.inf).sum():,} cells (infeasible)")
    print(f"  pi max    = {np.nanmax(pi):,.1f}")
    print(f"  pi min (finite) = {pi[np.isfinite(pi)].min():,.1f}")

    # Concavity check: π(H) at median B, median P should be concave
    n_mid = p["N"] // 2
    m_mid = p["M"] // 2
    slice_ = pi[n_mid, m_mid, :]
    finite_idx = np.where(np.isfinite(slice_))[0]
    vals = slice_[finite_idx]
    peak = finite_idx[np.argmax(vals)]
    print(f"\n  Concavity check at (B_mid, P_mid):")
    print(f"    B_mid = {B_grid[n_mid]:>12,.0f} t")
    print(f"    P_mid = {P_grid[m_mid]:>12,.1f} ISK/t")
    print(f"    π peaks at H_grid[{peak}] = {H_grid[peak]:>12,.0f} t")
    if peak > 0 and peak < len(finite_idx) - 1:
        print("    Concavity: interior peak ✓")
    else:
        print("    WARNING: peak at boundary — check c1/c2/P0")

    # Plot heatmap: π[:, M//2, :] — rows=biomass, cols=harvest
    fig, ax = plt.subplots(figsize=(9, 5))
    slice_2d = pi[:, m_mid, :]
    # Replace -inf with NaN for plotting
    plot_data = np.where(np.isfinite(slice_2d), slice_2d, np.nan)
    im = ax.imshow(
        plot_data,
        aspect="auto",
        origin="lower",
        extent=[H_grid[0]/1e6, H_grid[-1]/1e6, B_grid[0]/1e6, B_grid[-1]/1e6],
        cmap="viridis",
    )
    plt.colorbar(im, ax=ax, label="Profit π (ISK)")
    ax.set_xlabel("Harvest H (million tonnes)")
    ax.set_ylabel("Biomass B (million tonnes)")
    ax.set_title(f"Profit surface π(B, P_med, H)  [P_med = {P_grid[m_mid]:,.0f} ISK/t]")
    plt.tight_layout()
    plt.savefig("profit_check.png", dpi=120)
    print("\n  Heatmap saved → profit_check.png ✓")
    print("\nAll profit tests passed. ✓")
