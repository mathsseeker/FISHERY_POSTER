"""
params.py — All fixed parameters for the Iceland Cod Stochastic Bioeconomic Model
===================================================================================
Single source of truth. Every other module imports from here.
Biological values: ICES (2025) cod.27.5a Table 8.
Economic values:   Mota (2020), Pizarro & Schwartz (2021).
"""

import numpy as np
from pathlib import Path

# ── Try to update I0 / validation catch from the generated CSV ───────────────
_CSV = Path(__file__).parent / "iceland_cod_model_data.csv"
_I0_default       = 1_068_566   # B4+ biomass 2024 (tonnes)  — ICES 2025
_CATCH_ICES_2025  = 203_822     # ICES advised catch 2025 (tonnes)

if _CSV.exists():
    try:
        import pandas as _pd
        _df = _pd.read_csv(_CSV)
        # Most recent year with a B4plus_t reading
        _row = _df.dropna(subset=["B4plus_t"]).sort_values("year").iloc[-1]
        _I0_default = float(_row["B4plus_t"])
        # Most recent catch (validation benchmark)
        _row_c = _df.dropna(subset=["catch_t"]).sort_values("year").iloc[-1]
        _CATCH_ICES_2025 = float(_row_c["catch_t"])
    except Exception:
        pass  # fall back to hard-coded values silently


PARAMS: dict = {
    # ── Biological parameters (Iceland cod, 4 age classes) ───────────────────
    "n":        4,                              # age classes: 1, 2, 3, 4+
    "tau":      0.5,                            # fishing at mid-year
    "w_s":      np.array([0.37, 0.56, 0.79, 2.04]),   # weight-at-age (kg) ICES 2025
    "gamma_s":  np.array([0.00, 0.36, 0.82, 1.00]),   # maturity-at-age    ICES 2025
    "m_s":      np.array([0.60, 0.40, 0.30, 0.20]),   # natural mortality  ICES 2025
    "q_s":      np.array([0.44, 1.00, 0.82, 0.30]),   # selectivity        ICES 2025

    # Stochastic Ricker recruitment (fit to ICES SSB-recruitment series)
    "r1":       15.44,      # Ricker alpha  (scale)
    "r2":       0.686,      # Ricker beta   (density-dependence)
    "sigma_R":  0.30,       # std dev of log-recruitment shock

    # Carrying capacity proxy  (1.5 × peak B4+ of 1 229 426 t observed in 2016)
    "I_max":    1_844_319,  # tonnes

    # ── Economic parameters ──────────────────────────────────────────────────
    "nu_P":     0.02,       # log-price drift  (Pizarro & Schwartz 2021)
    "sigma_P":  0.16,       # log-price volatility
    "c1":       0.20,       # cost scale  — schooling fishery (Mota 2020)
    "c2":       1.27,       # cost curvature — MOST SENSITIVE PARAMETER
    "r_disc":   0.02,       # annual risk-adjusted discount rate
    "beta":     1 / 1.02,   # discount factor  = 1 / (1 + r_disc)

    # Initial / current state (ICES 2025)
    "I0":       _I0_default,            # B4+ biomass 2024  (tonnes)
    "P0":       120_000.0,              # ex-vessel price   (ISK/tonne, approx.)
    "catch_ices_2025": _CATCH_ICES_2025,  # ICES advised catch benchmark

    # ── ICES reference points ────────────────────────────────────────────────
    "B_lim":         125_000,   # biological limit reference point (ICES 2025)
    "MSY_Btrigger":  265_000,   # MSY Btrigger (ICES 2025)
    "mgt_Btrigger":  220_000,   # Iceland management plan SSB trigger
    "HR_MSY":        0.22,      # target harvest rate at MSY

    # ── VFI grid parameters (MacBook Air calibrated) ─────────────────────────
    "N":  50,   # biomass grid points
    "M":  50,   # price grid points
    "J":  80,   # harvest policy grid points
    "K":  25,   # Gauss-Hermite quadrature nodes for recruitment shock

    # ── Viability parameters (Briton et al. 2020) — used in viability.py ────
    "wage_min":     5_000_000,  # minimum annual wage per crew member (ISK)
    "fixed_cost":   50_000_000, # annual fixed operating cost (ISK)
    "depreciation": 20_000_000, # annual capital depreciation (ISK)
    "crew_share":   0.25,       # share of revenue going to crew wages
    "FTE_crew":     500,        # full-time-equivalent crew members
}


if __name__ == "__main__":
    print("=" * 60)
    print("PARAMS — Iceland Cod Bioeconomic Model")
    print("=" * 60)
    for k, v in PARAMS.items():
        if isinstance(v, np.ndarray):
            print(f"  {k:20s}: {v}")
        elif isinstance(v, float):
            print(f"  {k:20s}: {v:.6g}")
        else:
            print(f"  {k:20s}: {v}")
    print()
    print(f"  Derived: I_max = {PARAMS['I_max']:>12,.0f} t")
    print(f"  Derived: beta  = {PARAMS['beta']:.6f}")
    print(f"  Derived: I0    = {PARAMS['I0']:>12,.0f} t")
    assert PARAMS["c2"] > 1.0,          "c2 must be > 1 (cost curvature)"
    assert len(PARAMS["w_s"])     == 4,  "w_s must have 4 elements"
    assert len(PARAMS["gamma_s"]) == 4,  "gamma_s must have 4 elements"
    assert len(PARAMS["m_s"])     == 4,  "m_s must have 4 elements"
    assert len(PARAMS["q_s"])     == 4,  "q_s must have 4 elements"
    print("\n  All assertions passed. ✓")
