"""
params.py — All fixed parameters for the Iceland Cod Stochastic Bioeconomic Model
===================================================================================
Single source of truth. Every other module imports from here.
Biological values: ICES (2025) cod.27.5a Table 4, Table 5, Table 8.
Economic values:   Mota (2020), Pizarro & Schwartz (2021).

AUDIT LOG (2026-04-11):
  m_s   — corrected from [0.60,0.40,0.30,0.20] to [0.20,0.20,0.20,0.20]
           Source: ICES 2025 Table 5 "Natural mortality is 0.2 at all ages and years."
           Previous values were from Atlantic horse mackerel (Mota 2020) — wrong stock.
  r2    — updated 3.5569e-6 → 2.9673e-6 from full-series Ricker refit (1955-2024, n=69).
  sigma_R — updated 0.30 → 0.33 from std of log-recruitment residuals (full series).
  I_max — updated to actual historical peak B4+ (1955, Table 8) from 1.5×2016 peak.
  w_s, gamma_s, q_s — PROVISIONAL; detailed age tables not published in advice sheet PDF.
                      Source: ICES NWWG 2025 (https://doi.org/10.17895/ices.pub.29086181).
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
    "w_s":      np.array([0.37, 0.56, 0.79, 2.04]),
    # Units: kg/fish. The age-class vector x is in thousands of fish.
    # dot(w_s_kg, x_thousands) = tonnes, consistent with B_grid and H_grid.
    # recruit() returns R in thousands → x[0] stays in thousands each period.
    # Source: Mota (2020) Table A1 — w_s in kg/individual.
    # PROVISIONAL — exact age-weight vectors not published in ICES 2025 advice PDF;
    # full table in NWWG 2025 report (doi:10.17895/ices.pub.29086181).
    # Table 5 notes weights "lower than long-term average" in 2023-2024.

    "gamma_s":  np.array([0.00, 0.36, 0.82, 1.00]),
    # maturity-at-age ages 1-4+. PROVISIONAL — consistent with ICES 2025 Table 5
    # (maturity from IS-SMB/IS-SMH bottom-trawl surveys); exact vectors in NWWG 2025.

    "m_s":      np.array([0.20, 0.20, 0.20, 0.20]),
    # natural mortality ages 1-4+. SOURCE: ICES 2025 Table 5 (authoritative):
    # "Natural mortality is 0.2 at all ages and years."
    # Previous values [0.60,0.40,0.30,0.20] were from Atlantic horse mackerel — WRONG.

    "q_s":      np.array([0.44, 1.00, 0.82, 0.30]),
    # selectivity (exploitation pattern) ages 1-4+. PROVISIONAL — not published in
    # advice sheet; consistent with a dome-shaped pattern where age-2 is fully
    # selected and old fish (4+) partially escape via depth refuge. Verify against
    # NWWG 2025 separable catch-at-age model output.

    # Stochastic Ricker recruitment — R = r1*SSB*exp(-r2*SSB)
    # Fit: scipy.optimize.curve_fit on ICES 2025 Table 8 central estimates,
    # all 69 consecutive SSB_t → R_{t+1} pairs (1955-2024).
    # Units: SSB in tonnes, R in thousands of fish. R² = -0.05 (typical for cod —
    # stock-recruitment relationship is weak; stochastic noise dominates).
    "r1":       1.5140,     # Ricker alpha  — full-series refit (2026-04-11)
    "r2":       2.9673e-6,  # Ricker beta   — full-series refit (2026-04-11)
    "sigma_R":  0.33,       # std of log-recruitment residuals — full series

    # Carrying capacity: central value for Icelandic cod.
    # Historical B4+ maximum ~1.2–1.5 M t (1950s warm period, ICES Table 8).
    # 1 300 000 t places I0 ≈ 972 000 at 75% of capacity — a healthy, harvestable state.
    "I_max":    1_300_000,  # tonnes — calibrated carrying capacity (grid ceiling)

    # ── VFI grid boundary overrides ─────────────────────────────────────────────
    # These decouple the grid bounds from I_max so each can be set independently.
    "B_grid_lo":   50_000,   # biomass grid lower bound (t) — hard floor above B_lim
    "H_grid_max": 250_000,   # harvest grid upper bound (t) — reverted to 25% above ICES TAC

    # ── Economic parameters ──────────────────────────────────────────────────
    "nu_P":     0.02,       # log-price drift  (Pizarro & Schwartz 2021)
    "sigma_P":  0.18,       # log-price volatility (recalibrated)

    # ── Cost parameters — Mota (2020) power-law: π = P·H − c1·H^c2  ────────────
    # Calibration (2026-04-12, EUR/tonne units, Path A):
    #   Set H*_static = H_target = 200,000 t at P_target = 5,500 EUR/t
    #   FOC: dπ/dH = P − c1·c2·H^(c2−1) = 0
    #   c1 = P_target / (c2 × H_target^(c2−1))
    #      = 5500 / (1.27 × 200000^0.27)
    #      = 5500 / (1.27 × 27.003) = 160.428
    #   Implied margin at H*: 1 − 1/c2 = 1 − 1/1.27 = 21.3% (passes Check B ≥ 20%)
    "c1":       160.428,    # cost scale (EUR·tonne^(−1.27))
    "c2":       1.27,       # cost curvature exponent (Mota 2020 benchmark)

    "r_disc":   0.08,       # annual risk-adjusted discount rate (commercial fishery)
    "beta":     1 / 1.08,   # discount factor  = 1 / (1 + r_disc)

    # Initial / current state
    # I0: B4+ biomass. Hard-coded default = 2024 confirmed value from ICES Table 8.
    # If iceland_cod_model_data.csv exists, the CSV-reader above overrides this with
    # the most recent row (2025 forecast = 972,145 t, ICES Table 8).
    # I0 2024 (confirmed): 1,068,566 t  | I0 2025 (forecast): 972,145 t
    "I0":       _I0_default,
    "P0":       5_500,                  # ex-vessel price (EUR/tonne) — recalibrated
    # Price grid override: covers [2500, 12100] EUR/tonne (realistic range ±1 SD over 10 yr)
    "P_grid_lo": 2_500,    # EUR/tonne — price grid lower bound
    "P_grid_hi": 12_100,   # EUR/tonne — price grid upper bound
    "catch_ices_2025": _CATCH_ICES_2025,  # ICES 2025/2026 advice: ≤ 203,822 t

    # ── ICES reference points — all verified against ICES 2025 Table 4 ─────────
    "B_lim":         125_000,   # Blim = Bloss (ICES 2010a / 2025 Table 4) ✓
    "B_PA":          160_000,   # Bpa = Blim × exp(1.645×0.15) (ICES 2016) ✓
    "MSY_Btrigger":  265_000,   # MSY Btrigger, lower 5th pct SSB@HRMSY ✓
    "mgt_Btrigger":  220_000,   # Iceland management plan SSB trigger ✓
    "HR_MSY":        0.22,      # HRMSY = proportion of B4+  ✓
    "HR_mgt":        0.20,      # Iceland management plan HR ✓

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
    assert PARAMS["c2"] > 1.0,          "c2 must be > 1 (Mota power-law curvature)"
    assert len(PARAMS["w_s"])     == 4,  "w_s must have 4 elements"
    assert len(PARAMS["gamma_s"]) == 4,  "gamma_s must have 4 elements"
    assert len(PARAMS["m_s"])     == 4,  "m_s must have 4 elements"
    assert len(PARAMS["q_s"])     == 4,  "q_s must have 4 elements"
    print("\n  All assertions passed. ✓")
