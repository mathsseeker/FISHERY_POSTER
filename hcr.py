"""
hcr.py — Extract piecewise-linear Harvest Control Rule from H*(B,P)
=====================================================================
Fits a 3-piece HCR to each price column of the VFI policy surface:

    H = 0                           if B <= B_zero
    H = slope * (B - B_zero)        if B_zero < B < B_zero + H_max/slope
    H = H_max                       if B >= B_zero + H_max/slope
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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


# ─────────────────────────────────────────────────────────────────────────────
# Piecewise model
# ─────────────────────────────────────────────────────────────────────────────

def _hcr_piecewise(B, B_zero, slope, H_max):
    """Evaluate the 3-piece HCR at biomass points B."""
    B = np.asarray(B, dtype=float)
    # Ramp ends when slope*(B-B_zero) = H_max  →  B_ramp = B_zero + H_max/slope
    with np.errstate(divide="ignore", invalid="ignore"):
        B_ramp = np.where(slope > 1e-12, B_zero + H_max / slope, np.inf)
    return np.where(B <= B_zero, 0.0,
           np.where(B <= B_ramp,  slope * (B - B_zero), H_max))


# ─────────────────────────────────────────────────────────────────────────────
# Function 1 — fit HCR to every price column
# ─────────────────────────────────────────────────────────────────────────────

def extract_hcr(H_star, B_grid, P_grid, p=PARAMS):
    """
    Extract 3-piece HCR parameters directly from the VFI policy surface.

    Reads breakpoints from the data rather than fitting them with a
    gradient-based optimizer (curve_fit fails on non-differentiable
    piecewise functions — it pushes H_max to infinity so the ceiling
    never appears, and misplaces B_zero causing spurious non-zero dots
    below the closure threshold).

    Method
    ------
    1. H_max  — median of H*(B) for B ≥ B_MSY  (actual plateau value)
    2. B_zero — midpoint before the first B where H* > tol  (closure point)
    3. slope  — OLS on the ramp region only: B_zero < B < B_ramp

    Returns
    -------
    dict with keys: B_zero [M], slope [M], H_max [M]
    """
    M      = len(P_grid)
    I_max  = p["I_max"]
    B_MSY  = p["MSY_Btrigger"]   # plateau kicks in here (265 kt)
    tol    = 1.0                  # tonnes — treat H* < tol as zero

    B_zero_arr = np.zeros(M)
    slope_arr  = np.zeros(M)
    H_max_arr  = np.zeros(M)

    plateau_idx = np.where(B_grid >= B_MSY)[0]   # indices in plateau region

    for m in range(M):
        H_col = H_star[:, m].copy()
        nz    = np.where(H_col > tol)[0]

        # Price so low no harvest is ever optimal → trivial rule
        if len(nz) == 0:
            B_zero_arr[m] = I_max
            slope_arr[m]  = 0.0
            H_max_arr[m]  = 0.0
            continue

        # ── Step 1: H_max from plateau ─────────────────────────────────────
        if len(plateau_idx) > 0:
            H_plateau = H_col[plateau_idx]
            H_plateau_pos = H_plateau[H_plateau > tol]
            H_max = float(np.median(H_plateau_pos)) if len(H_plateau_pos) > 0 \
                    else float(H_col[nz].max())
        else:
            H_max = float(H_col[nz].max())

        # ── Step 2: B_zero from first non-zero point ───────────────────────
        i_first = nz[0]
        B_zero  = float((B_grid[max(0, i_first - 1)] + B_grid[i_first]) / 2.0)

        # ── Step 3: slope via OLS on ramp region ───────────────────────────
        # Ramp: B_zero < B, and H* has not yet plateaued (H* < 0.95 * H_max)
        ramp_mask = (B_grid > B_zero) & (H_col > tol) & (H_col < 0.95 * H_max)
        ramp_idx  = np.where(ramp_mask)[0]

        if len(ramp_idx) >= 2:
            X = B_grid[ramp_idx] - B_zero   # distance from closure threshold
            Y = H_col[ramp_idx]
            # OLS through origin: slope = (X·Y) / (X·X)
            slope = float(np.dot(X, Y) / np.dot(X, X))
            slope = max(0.0, min(1.0, slope))
        elif H_max > tol:
            # Fallback: straight line from B_zero to right edge of grid
            span  = max(B_grid[-1] - B_zero, 1.0)
            slope = float(np.clip(H_max / span, 0.0, 1.0))
        else:
            slope = 0.0

        B_zero_arr[m] = B_zero
        slope_arr[m]  = slope
        H_max_arr[m]  = H_max

    return {"B_zero": B_zero_arr, "slope": slope_arr, "H_max": H_max_arr}


# ─────────────────────────────────────────────────────────────────────────────
# Function 2 — main poster figure
# ─────────────────────────────────────────────────────────────────────────────

def plot_hcr(hcr_params, H_star, B_grid, P_grid, p=PARAMS):
    """
    Scatter H*(B) + fitted HCR lines for 5 price levels.
    Marks B_lim and B_MSY.  Saves hcr_plot.png.
    """
    M     = len(P_grid)
    B_lim = p["B_lim"]
    B_MSY = p["MSY_Btrigger"]

    # 5 representative price indices (low → high)
    idx5  = np.linspace(0, M - 1, 5, dtype=int)
    cmap  = plt.cm.plasma(np.linspace(0.1, 0.85, 5))

    fig, ax = plt.subplots(figsize=(13, 7.5))
    B_fine  = np.linspace(B_grid[0], B_grid[-1], 1000)

    for rank, m in enumerate(idx5):
        P_label = f"{P_grid[m]/1_000:.0f}k EUR/t"
        col     = cmap[rank]

        # Scatter: VFI optimal policy
        ax.scatter(
            B_grid / 1e6, H_star[:, m] / 1e3,
            color=col, alpha=0.65, s=50, zorder=3,
        )

        # Fitted HCR curve
        B_z = hcr_params["B_zero"][m]
        sl  = hcr_params["slope"][m]
        Hm  = hcr_params["H_max"][m]
        H_fit = _hcr_piecewise(B_fine, B_z, sl, Hm)
        ax.plot(
            B_fine / 1e6, H_fit / 1e3,
            color=col, lw=2.5, linestyle="--",
            label=f"P = {P_label}",
        )

    # ── Reference verticals ────────────────────────────────────────────────
    ax.axvline(
        B_lim / 1e6, color="crimson", linestyle=":", lw=1.8,
        label=f"$B_{{lim}}$ = {B_lim/1e3:.0f} kt",
    )
    ax.axvline(
        B_MSY / 1e6, color="darkorange", linestyle=":", lw=1.8,
        label=f"$B_{{MSY}}$ = {B_MSY/1e3:.0f} kt",
    )

    # ── Label B_zero at median price ───────────────────────────────────────
    m_med      = M // 2
    B_zero_med = hcr_params["B_zero"][m_med]
    ax.axvline(
        B_zero_med / 1e6, color="black", linestyle="-.", lw=2.5,
        label=f"$B_{{zero}}$(med P) = {B_zero_med/1e3:.0f} kt",
    )
    # Annotation near bottom, to the right of the line
    ax.text(
        B_zero_med / 1e6 + 0.03, 0.08,
        f"$B_{{zero}}$ = {B_zero_med/1e3:.0f} kt",
        transform=ax.get_xaxis_transform(),
        fontsize=13, color="black", va="bottom",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", alpha=1.0),
    )

    # ── Cosmetics ──────────────────────────────────────────────────────────
    ax.set_xlabel("Biomass  B  (million tonnes)", fontsize=15)
    ax.set_ylabel("Optimal Harvest  H*  (thousand tonnes)", fontsize=15)
    ax.set_title(
        "Harvest Control Rule — Icelandic Cod\n"
        r"$H^*(B,P)$ from Value-Function Iteration  (VFI policy surface)",
        fontsize=16,
    )
    ax.tick_params(labelsize=13)
    ax.legend(fontsize=15, loc="upper right", ncol=2, framealpha=1.0, facecolor="white")
    ax.grid(True, alpha=0.25)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)

    plt.tight_layout()
    plt.savefig("hcr_plot.png", dpi=180)
    plt.close()
    print("  Saved: hcr_plot.png")


# ─────────────────────────────────────────────────────────────────────────────
# Function 3 — summary table
# ─────────────────────────────────────────────────────────────────────────────

def print_hcr_table(hcr_params, P_grid):
    """Print HCR parameters at 5 representative price levels."""
    M    = len(P_grid)
    idx5 = np.linspace(0, M - 1, 5, dtype=int)

    header = (
        f"\n{'Price (ISK/t)':>16} | {'B_zero (t)':>12} | "
        f"{'slope':>8} | {'H_max (t)':>12}"
    )
    print("\n" + "=" * 57)
    print("HCR Parameters — 5 Representative Price Levels")
    print("=" * 57)
    print(header)
    print("-" * 57)
    for m in idx5:
        print(
            f"{P_grid[m]:>16,.0f} | "
            f"{hcr_params['B_zero'][m]:>12,.0f} | "
            f"{hcr_params['slope'][m]:>8.4f} | "
            f"{hcr_params['H_max'][m]:>12,.0f}"
        )
    print("=" * 57)


# ─────────────────────────────────────────────────────────────────────────────
# __main__
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import os
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    print("=" * 60)
    print("hcr.py — Harvest Control Rule extraction")
    print("=" * 60)

    B_grid = np.load("B_grid.npy")
    P_grid = np.load("P_grid.npy")
    H_star = np.load("H_star.npy")

    print(f"  Loaded H_star {H_star.shape},  "
          f"B_grid [{B_grid.min():,.0f}, {B_grid.max():,.0f}] t,  "
          f"P_grid [{P_grid.min():,.0f}, {P_grid.max():,.0f}] ISK/t")

    hcr_params = extract_hcr(H_star, B_grid, P_grid)
    plot_hcr(hcr_params, H_star, B_grid, P_grid)
    print_hcr_table(hcr_params, P_grid)

    # Quick sanity check
    m_med = len(P_grid) // 2
    print(f"\n  B_zero at median price ({P_grid[m_med]:,.0f} ISK/t): "
          f"{hcr_params['B_zero'][m_med]:,.0f} t")
    print(f"  ICES mgt_Btrigger: {PARAMS['mgt_Btrigger']:,.0f} t")
    print("\nHCR extraction complete. ✓")
