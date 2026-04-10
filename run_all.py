"""
run_all.py — Master runner for the Iceland Cod Bioeconomic Model
================================================================
Pipeline:
  1.  Read I0 from iceland_cod_model_data.csv (most recent B4plus_t row)
      or fall back to PARAMS default.
  2.  Run VFI (vfi.solve_vfi) → saves .npy files.
  3.  Run HCR extraction + plot (hcr.extract_hcr, hcr.plot_hcr).
  4.  Print final summary table.
"""

import os
import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator

# ── ensure we run from the project directory ───────────────────────────────
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from params import PARAMS
import vfi
import hcr as hcr_mod


# ─────────────────────────────────────────────────────────────────────────────
# Step 1 — resolve I0
# ─────────────────────────────────────────────────────────────────────────────

def get_I0(p: dict) -> float:
    """
    Return the most recent B4plus_t from iceland_cod_model_data.csv,
    or fall back to p['I0'] if the CSV is absent / unreadable.
    """
    csv_path = "iceland_cod_model_data.csv"
    if os.path.exists(csv_path):
        try:
            df   = pd.read_csv(csv_path)
            row  = df.dropna(subset=["B4plus_t"]).sort_values("year").iloc[-1]
            I0   = float(row["B4plus_t"])
            year = int(row["year"])
            print(f"  [CSV]  I0 = {I0:,.0f} t  (year {year})")
            return I0
        except Exception as e:
            print(f"  [CSV]  Could not parse CSV ({e}); using PARAMS default.")
    else:
        print("  [CSV]  iceland_cod_model_data.csv not found; using PARAMS default.")
    return float(p["I0"])


# ─────────────────────────────────────────────────────────────────────────────
# Step 4 — summary
# ─────────────────────────────────────────────────────────────────────────────

def _nearest_index(grid: np.ndarray, value: float) -> int:
    return int(np.argmin(np.abs(grid - value)))


def _interpolate_value(V_star, B_grid, P_grid, B_val, P_val) -> float:
    """Bilinear interpolation of V_star at (B_val, P_val)."""
    interp = RegularGridInterpolator(
        (B_grid, P_grid), V_star,
        method="linear", bounds_error=False, fill_value=None,
    )
    return float(interp([[B_val, P_val]])[0])


def _compute_V_fixed_HR(I0, P0, p) -> float:
    """
    Approximate value if always harvesting at HR_MSY * I0  (perpetuity).

    π_fixed = P0 * H_fixed  -  c1 * H_fixed^c2
    V_fixed = π_fixed / (1 - β)           [geometric series, constant π]

    This is a conservative lower bound (it ignores stock dynamics and
    price drift, but serves as a simple benchmark for the poster).
    """
    H_fixed = p["HR_MSY"] * I0
    c1, c2  = p["c1"], p["c2"]
    beta    = p["beta"]
    pi_fixed = P0 * H_fixed - c1 * (H_fixed ** c2)
    return pi_fixed / (1.0 - beta)


def print_summary(V_star, H_star, B_grid, P_grid, hcr_params, I0, p):
    """Print the final four-line summary table."""
    P0        = p["P0"]
    beta      = p["beta"]
    ICES_adv  = p["catch_ices_2025"]
    mgt_Btrig = p["mgt_Btrigger"]

    n0 = _nearest_index(B_grid, I0)
    m0 = _nearest_index(P_grid, P0)

    H_opt   = H_star[n0, m0]
    V_opt   = _interpolate_value(V_star, B_grid, P_grid, I0, P0)
    V_fixed = _compute_V_fixed_HR(I0, P0, p)
    V_gap   = V_opt - V_fixed

    m_med         = len(P_grid) // 2
    B_zero_med    = hcr_params["B_zero"][m_med]

    print("\n" + "=" * 65)
    print("  FINAL SUMMARY  —  Iceland Cod Bioeconomic Model")
    print("=" * 65)
    print(f"  {'Metric':<40} {'Model':>12}  {'Benchmark':>12}")
    print("-" * 65)
    print(
        f"  {'Optimal H*(I0, P0)  [tonnes]':<40} "
        f"{H_opt:>12,.0f}  "
        f"{ICES_adv:>12,.0f}  ← ICES advice 2025"
    )
    print(
        f"  {'B_zero at median price  [tonnes]':<40} "
        f"{B_zero_med:>12,.0f}  "
        f"{mgt_Btrig:>12,.0f}  ← ICES mgt_Btrigger"
    )
    print(
        f"  {'V*(I0, P0)  [ISK]':<40} "
        f"{V_opt:>12,.0f}"
    )
    print(
        f"  {'Value gap  V_opt - V_fixed_HR  [ISK]':<40} "
        f"{V_gap:>12,.0f}"
    )
    print("=" * 65)
    print(f"\n  I0 used: {I0:,.0f} t   |   P0: {P0:,.0f} ISK/t   |   β: {beta:.4f}")
    print(f"  H_fixed_HR = HR_MSY × I0 = {p['HR_MSY']} × {I0:,.0f} = "
          f"{p['HR_MSY']*I0:,.0f} t")
    print(f"  V_fixed_HR (perpetuity approx.): {V_fixed:,.0f} ISK")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────────────────────────────────────

def main():
    p = dict(PARAMS)   # work on a copy so we can override I0

    print("=" * 65)
    print("  run_all.py — Iceland Cod Bioeconomic Model  (full pipeline)")
    print("=" * 65)

    # ── Step 1: I0 ────────────────────────────────────────────────────────
    print("\n[Step 1]  Resolving initial biomass I0 ...")
    I0   = get_I0(p)
    p["I0"] = I0

    # ── Step 2: VFI ───────────────────────────────────────────────────────
    print("\n[Step 2]  Running Value-Function Iteration ...")
    V_star, H_star, B_grid, P_grid = vfi.solve_vfi(p)

    # ── Step 3: HCR ───────────────────────────────────────────────────────
    print("\n[Step 3]  Extracting Harvest Control Rule ...")
    hcr_params = hcr_mod.extract_hcr(H_star, B_grid, P_grid, p)
    hcr_mod.plot_hcr(hcr_params, H_star, B_grid, P_grid, p)
    hcr_mod.print_hcr_table(hcr_params, P_grid)

    # ── Step 4: Summary ───────────────────────────────────────────────────
    print("\n[Step 4]  Final summary ...")
    print_summary(V_star, H_star, B_grid, P_grid, hcr_params, I0, p)

    print("Pipeline complete. ✓")


if __name__ == "__main__":
    main()
