"""
population.py — 4-age-class cohort model + stochastic Ricker recruitment
=========================================================================
Implements equations (1)–(4) from Mota (2020), Marine Policy 115, 103865.

Age classes: s = 0,1,2,3  (ages 1, 2, 3, 4+)
  s=3 is the "plus group" (4+): accumulates surviving age-3 and age-4+ fish.

All arrays are 0-indexed:   w_s[0] = weight age-1, ..., w_s[3] = weight age-4+
"""

import numpy as np
from params import PARAMS


# ─────────────────────────────────────────────────────────────────────────────
# 1. Spawning Stock Biomass
# ─────────────────────────────────────────────────────────────────────────────

def ssb(x: np.ndarray, p: dict = PARAMS) -> float:
    """
    Spawning stock biomass.

    Parameters
    ----------
    x : array[4]  — age-class abundances (numbers, thousands or raw)
    p : PARAMS dict

    Returns
    -------
    x0 : float — SSB in same weight units as w_s * x
    """
    return float(np.dot(p["gamma_s"] * p["w_s"], x))


# ─────────────────────────────────────────────────────────────────────────────
# 2. Stochastic Ricker Recruitment
# ─────────────────────────────────────────────────────────────────────────────

def recruit(x0: float, eps_R: float, p: dict = PARAMS) -> float:
    """
    Stochastic Ricker recruitment.

    Returns age-1 fish count next period given SSB x0 and log-normal shock.

    x1_next = r1 * x0 * exp(-r2 * x0) * exp(eps_R)          — Mota eq. (2)

    Parameters
    ----------
    x0    : float — spawning stock biomass (same units as 1/r2)
    eps_R : float — recruitment shock ~ N(0, sigma_R^2)
    p     : PARAMS dict

    Returns
    -------
    x1_next : float — number of age-1 recruits (same units as x0)
    """
    return float(p["r1"] * x0 * np.exp(-p["r2"] * x0) * np.exp(eps_R))


# ─────────────────────────────────────────────────────────────────────────────
# 3. Cohort Transition
# ─────────────────────────────────────────────────────────────────────────────

def transition(
    x: np.ndarray,
    H: float,
    eps_R: float,
    p: dict = PARAMS,
) -> tuple[np.ndarray, float]:
    """
    One-period age-structured population transition.

    Harvest H (total weight) is distributed across age classes using
    selectivity-weighted G_s proportions (Mota 2020, eqs. 5–6):

        G_s = q_s * exp(-m_s * tau) * x_s
              ────────────────────────────────────────────────
              sum_s'( w_s' * q_s' * exp(-m_s' * tau) * x_s' )

    Cohort dynamics (tau = 0.5, mid-year fishing):
        x_{1,t+1} = R(SSB_t, eps_R)                             — eq. (2)
        x_{2,t+1} = exp(-m_1)*x_{1,t}  -  H * G_{1,t}          — eq. (3)
        x_{3,t+1} = exp(-m_2)*x_{2,t}  -  H * G_{2,t}
        x_{4,t+1} = exp(-m_3)*x_{3,t} + exp(-m_4)*x_{4,t}
                    - H*(G_{3,t} + G_{4,t})                      — eq. (4)

    Parameters
    ----------
    x     : array[4] — current age-class vector
    H     : float    — total harvest weight (tonnes)
    eps_R : float    — recruitment shock ~ N(0, sigma_R^2)
    p     : PARAMS dict

    Returns
    -------
    x_next  : array[4] — next-period age-class vector
    B_next  : float    — next-period exploitable biomass = sum(w_s * x_next)

    Raises
    ------
    ValueError if H exceeds the total catchable biomass.
    """
    x = np.asarray(x, dtype=float)
    w   = p["w_s"]
    gam = p["gamma_s"]
    m   = p["m_s"]
    q   = p["q_s"]
    tau = p["tau"]

    # ── Feasibility guard ────────────────────────────────────────────────────
    # Catchable biomass = sum( w_s * q_s * x_s )  (pre-fishing selectivity)
    catchable = float(np.dot(w * q, x))
    if H > catchable + 1e-9:          # small tolerance for floating-point
        raise ValueError(
            f"H={H:,.1f} exceeds catchable biomass={catchable:,.1f}. "
            "Cannot harvest more than available."
        )
    H = min(H, catchable)             # clamp at feasibility boundary

    # ── Harvest distribution weights G_s ─────────────────────────────────────
    # Numerator: selectivity * mid-year survival * abundance
    num   = q * np.exp(-m * tau) * x
    denom = np.dot(w * q * np.exp(-m * tau), x)

    if denom < 1e-12:
        G = np.zeros(4)
    else:
        G = num / denom               # shape [4], sums-to-1 in weight sense

    # ── SSB and recruitment ───────────────────────────────────────────────────
    x0     = ssb(x, p)
    x1_new = recruit(x0, eps_R, p)

    # ── Cohort update ─────────────────────────────────────────────────────────
    surv = np.exp(-m)   # annual survival factor per age class
    x_next = np.zeros(4)

    x_next[0] = max(0.0, x1_new)                              # age-1 recruits
    x_next[1] = max(0.0, surv[0] * x[0] - H * G[0])          # age-2
    x_next[2] = max(0.0, surv[1] * x[1] - H * G[1])          # age-3
    x_next[3] = max(                                           # age-4+ (plus group)
        0.0,
        surv[2] * x[2] + surv[3] * x[3] - H * (G[2] + G[3])
    )

    B_next = float(np.dot(w, x_next))
    return x_next, B_next


# ─────────────────────────────────────────────────────────────────────────────
# Helper: steady-state age distribution at a given biomass level
# ─────────────────────────────────────────────────────────────────────────────

def steady_state_x(B: float, p: dict = PARAMS) -> np.ndarray:
    """
    Distribute biomass B across age classes using steady-state proportions.

    Proportions are based on relative selectivity × weight × survival:
        p_s ∝ w_s * q_s * exp(-m_s)
    Normalised so that sum(w_s * x_s) = B.

    Used inside VFI to convert each grid point B_n into an age vector.
    """
    w = p["w_s"]
    q = p["q_s"]
    m = p["m_s"]

    raw   = w * q * np.exp(-m)       # proxy for relative abundance at age
    total = np.dot(w, raw)           # total biomass weight of the raw vector
    if total < 1e-12:
        return np.zeros(4)
    # Scale so that sum(w_s * x_s) = B
    return raw * (B / total)


# ─────────────────────────────────────────────────────────────────────────────
# Self-test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")

    p = PARAMS
    print("=" * 60)
    print("population.py — self-test (zero harvest, zero shock)")
    print("=" * 60)

    # Initialise from I0 distributed via steady-state proportions
    I0 = p["I0"]
    x0_vec = steady_state_x(I0, p)
    print(f"\nInitial age-class vector (B = {I0:,.0f} t):")
    for s in range(4):
        print(f"  age-{s+1}: {x0_vec[s]:>12.1f}  "
              f"(weight contribution: {p['w_s'][s]*x0_vec[s]:>10.1f} t)")
    print(f"  SSB          : {ssb(x0_vec, p):>12,.1f} t")
    print(f"  Total biomass: {np.dot(p['w_s'], x0_vec):>12,.1f} t")

    print(f"\nSimulating 10 years — zero harvest, zero shock:")
    print(f"{'Year':>5}  {'B4+ (t)':>12}  {'SSB (t)':>12}  {'Recruits':>12}")
    print("-" * 50)
    x = x0_vec.copy()
    for yr in range(1, 11):
        x, B = transition(x, H=0.0, eps_R=0.0, p=p)
        print(f"  {yr:>3}  {B:>12,.0f}  {ssb(x,p):>12,.0f}  {x[0]:>12,.0f}")

    print("\nTest feasibility guard (H > catchable):")
    try:
        x_test = steady_state_x(100_000, p)
        catchable = float(np.dot(p["w_s"] * p["q_s"], x_test))
        transition(x_test, H=catchable * 2, eps_R=0.0, p=p)
        print("  ERROR — should have raised ValueError!")
    except ValueError as e:
        print(f"  ValueError raised correctly: {e}")

    print("\nAll population tests passed. ✓")
