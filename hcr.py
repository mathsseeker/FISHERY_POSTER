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
from scipy.optimize import curve_fit

from params import PARAMS


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
    Fit the 3-piece HCR to H_star[:,m] for each price index m.

    Bounds:
        B_zero  in [B_lim,  I_max]        (allow VFI threshold anywhere)
        slope   in [0,      1]
        H_max   in [0,      0.4*I_max]

    Returns
    -------
    dict with keys: B_zero [M], slope [M], H_max [M]
    """
    M     = len(P_grid)
    I_max = p["I_max"]
    B_lim = p["B_lim"]

    B_zero_arr = np.zeros(M)
    slope_arr  = np.zeros(M)
    H_max_arr  = np.zeros(M)

    lower = [B_lim,  0.0, 0.0        ]
    upper = [I_max,  1.0, 0.4 * I_max]

    for m in range(M):
        H_col = H_star[:, m].copy()

        nz = np.where(H_col > 0)[0]

        # Price so low no harvest is ever optimal → trivial rule
        if len(nz) == 0:
            B_zero_arr[m] = I_max
            slope_arr[m]  = 0.0
            H_max_arr[m]  = 0.0
            continue

        # Initial guesses from data
        i_first     = nz[0]
        B_zero_g    = (B_grid[max(0, i_first - 1)] + B_grid[i_first]) / 2.0
        B_zero_g    = float(np.clip(B_zero_g, B_lim + 1, I_max - 1))
        H_max_g     = float(H_col[nz].max())
        span        = max(B_grid[-1] - B_zero_g, 1.0)
        slope_g     = float(np.clip(H_max_g / span, 0.0, 1.0))

        try:
            popt, _ = curve_fit(
                _hcr_piecewise, B_grid, H_col,
                p0=[B_zero_g, slope_g, H_max_g],
                bounds=(lower, upper),
                maxfev=20_000,
            )
            B_zero_arr[m] = popt[0]
            slope_arr[m]  = popt[1]
            H_max_arr[m]  = popt[2]
        except RuntimeError:
            # curve_fit failed — use heuristic estimates
            B_zero_arr[m] = B_zero_g
            slope_arr[m]  = slope_g
            H_max_arr[m]  = H_max_g

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

    fig, ax = plt.subplots(figsize=(11, 6.5))
    B_fine  = np.linspace(B_grid[0], B_grid[-1], 1000)

    for rank, m in enumerate(idx5):
        P_label = f"{P_grid[m]/1_000:.0f}k ISK/t"
        col     = cmap[rank]

        # Scatter: VFI optimal policy
        ax.scatter(
            B_grid / 1e6, H_star[:, m] / 1e3,
            color=col, alpha=0.65, s=30, zorder=3,
        )

        # Fitted HCR curve
        B_z = hcr_params["B_zero"][m]
        sl  = hcr_params["slope"][m]
        Hm  = hcr_params["H_max"][m]
        H_fit = _hcr_piecewise(B_fine, B_z, sl, Hm)
        ax.plot(
            B_fine / 1e6, H_fit / 1e3,
            color=col, lw=2.0, linestyle="--",
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
        B_zero_med / 1e6, color="black", linestyle="-.", lw=1.4,
        label=f"$B_{{zero}}$(med P) = {B_zero_med/1e3:.0f} kt",
    )
    # y position based on actual data range (computed after scatter/plot calls above)
    H_nonzero = H_star[H_star > 0]
    y_ref = (H_nonzero.max() / 1e3 * 0.55) if len(H_nonzero) > 0 else 50.0
    ax.annotate(
        f"$B_{{zero}}$\n{B_zero_med/1e3:.0f} kt",
        xy=(B_zero_med / 1e6, y_ref),
        xytext=(B_zero_med / 1e6 - 0.20, y_ref * 1.15),
        fontsize=9, color="black",
        arrowprops=dict(arrowstyle="->", color="black", lw=1.0),
    )

    # ── Cosmetics ──────────────────────────────────────────────────────────
    ax.set_xlabel("Biomass  B  (million tonnes)", fontsize=12)
    ax.set_ylabel("Optimal Harvest  H*  (thousand tonnes)", fontsize=12)
    ax.set_title(
        "Harvest Control Rule — Icelandic Cod\n"
        r"$H^*(B,P)$ from Value-Function Iteration  (VFI policy surface)",
        fontsize=13,
    )
    ax.legend(fontsize=9, loc="upper left", framealpha=0.85)
    ax.grid(True, alpha=0.25)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)

    plt.tight_layout()
    plt.savefig("hcr_plot.png", dpi=150)
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
