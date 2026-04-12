"""
vfi.py — Value-Function Iteration engine
=========================================
Bellman equation (Pizarro & Schwartz 2021, eq. 18):

    V(I_n, P_m) = max_j { π[n,m,j]  +  β · E[V(I_n', P_m')] }

Expectations are over:
  1. K=25 Gauss-Hermite quadrature nodes for recruitment shock ε_R ~ N(0,σ_R²)
  2. M×M Markov price transition matrix T_P

VFI iterates until max|V_new - V_old| < 1e-6  (max 2000 iterations).
Results are saved to .npy files so hcr.py can reload without re-running VFI.
"""

import time
import numpy as np
from scipy.interpolate import RegularGridInterpolator

from params     import PARAMS
from price      import build_price_chain
from profit     import build_grids, build_profit_tensor
from population import steady_state_x, transition

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
# Main solver
# ─────────────────────────────────────────────────────────────────────────────

def solve_vfi(p: dict = PARAMS):
    """
    Run Value-Function Iteration.

    Returns
    -------
    V_star : array[N, M]   — optimal value function
    H_star : array[N, M]   — optimal harvest policy (tonnes)
    B_grid : array[N]
    P_grid : array[M]
    """
    t0 = time.time()

    # ── Grids ────────────────────────────────────────────────────────────────
    P_grid, T_P     = build_price_chain(p)
    B_grid, H_grid  = build_grids(p, P_grid)

    N = len(B_grid)
    M = len(P_grid)
    J = len(H_grid)
    K = p["K"]
    beta    = p["beta"]
    sigma_R = p["sigma_R"]

    # ── Gauss-Hermite quadrature for ε_R ~ N(0, σ_R²) ───────────────────────
    # numpy.polynomial.hermite.hermgauss returns (nodes, weights) for the
    # weight function exp(-x²).  Transform: ε_k = node_k * sqrt(2) * σ_R
    # normalised weights sum to sqrt(π)  →  divide by sqrt(π) for probability.
    gh_nodes, gh_weights = np.polynomial.hermite.hermgauss(K)
    eps_nodes   = gh_nodes * np.sqrt(2.0) * sigma_R   # shape [K]
    eps_weights = gh_weights / np.sqrt(np.pi)          # shape [K], sum=1

    # ── Profit tensor  π[N, M, J] ────────────────────────────────────────────
    print("Building profit tensor ...", flush=True)
    pi_tensor = build_profit_tensor(B_grid, P_grid, H_grid, p)

    # ── Steady-state age vectors at each biomass grid point ──────────────────
    # x_grid[n] = age-class vector when total biomass = B_grid[n]
    x_grid = np.array([steady_state_x(B, p) for B in B_grid])  # [N, 4]

    # ── Precompute next-period biomass for every (n, j, k) triple ────────────
    # This is the most expensive precomputation but done once before the VFI.
    # Shape: B_next[n, j, k]  =  B_grid[n] → harvest H_grid[j] → shock eps_k
    print("Precomputing next-period biomass array B_next[N, J, K] ...", flush=True)
    B_next = np.zeros((N, J, K))
    for n in range(N):
        x_n = x_grid[n]
        catchable_n = float(np.dot(p["w_s"] * p["q_s"], x_n))
        for j in range(J):
            H_j = H_grid[j]
            if H_j > catchable_n + 1e-9:
                # Infeasible — fill with a small value; π already -inf here
                B_next[n, j, :] = B_grid[0]
            else:
                for k in range(K):
                    _, B_nk = transition(x_n, H_j, eps_nodes[k], p)
                    B_next[n, j, k] = max(B_nk, B_grid[0])

    # Clip to grid bounds
    B_next = np.clip(B_next, B_grid[0], B_grid[-1])

    print(f"  B_next range: [{B_next.min():,.0f}, {B_next.max():,.0f}] t", flush=True)

    # ── Estimate runtime ─────────────────────────────────────────────────────
    # Rough benchmark: each iteration is O(N*M*J) + interpolation
    print(f"\nEstimated VFI runtime: ~3–10 min on MacBook Air M-series "
          f"(N={N}, M={M}, J={J}, K={K}).", flush=True)
    print("Starting VFI ...\n", flush=True)

    # ── Initialise value function ────────────────────────────────────────────
    V = np.zeros((N, M))

    tol      = 1e-6
    max_iter = 2000
    converged = False

    for it in range(1, max_iter + 1):

        # ── Interpolator for current V ────────────────────────────────────
        # RegularGridInterpolator expects axes (B_grid, P_grid)
        interp_V = RegularGridInterpolator(
            (B_grid, P_grid),
            V,
            method="linear",
            bounds_error=False,
            fill_value=None,   # extrapolates at boundaries
        )

        # ── Expected continuation value E[V(B_next, P_next)] ─────────────
        # Step 1: E_R[V(B_next_njk, P_m')] over quadrature nodes k
        #   For each (n, j, k), we need V(B_next[n,j,k], P_m) for all m.
        #   Then weight by eps_weights[k] and T_P[m, m'].

        # Build query points for the interpolator: (B_next[n,j,k], P_grid[m])
        # We need shape [N, J, K, M, 2] → flatten to [N*J*K*M, 2]

        # B_next: [N, J, K]  → [N, J, K, 1]
        B_q = B_next[:, :, :, np.newaxis]  # [N, J, K, 1]
        # P_grid: [M]         → [1, 1, 1, M]
        P_q = P_grid[np.newaxis, np.newaxis, np.newaxis, :]  # [1,1,1,M]

        # Broadcast to [N, J, K, M]
        B_pts = np.broadcast_to(B_q, (N, J, K, M)).reshape(-1)
        P_pts = np.broadcast_to(P_q, (N, J, K, M)).reshape(-1)

        pts = np.stack([B_pts, P_pts], axis=1)   # [N*J*K*M, 2]
        V_interp = interp_V(pts).reshape(N, J, K, M)  # [N, J, K, M]

        # Step 2: weight over quadrature nodes k  (eps_weights: [K])
        # EV_R[n, j, m] = sum_k eps_weights[k] * V_interp[n, j, k, m]
        EV_R = np.einsum("njkm,k->njm", V_interp, eps_weights)  # [N, J, M]

        # Step 3: weight over price transitions T_P[m, m']
        # EV[n, j, m] = sum_m' T_P[m, m'] * EV_R[n, j, m']
        # EV_R: [N, J, M]  T_P: [M, M]
        # EV[n, j, m] = EV_R[n, j, :] @ T_P[m, :]^T  = EV_R[n,j,:] @ T_P.T[:,m]
        # Efficient: EV = EV_R @ T_P.T   but shapes: [N,J,M] @ [M,M] → [N,J,M]
        EV = EV_R @ T_P.T   # [N, J, M]  (T_P[m,m'] → sum over m' first)
        # Actually: EV[n,j,m] = sum_{m'} EV_R[n,j,m'] * T_P[m, m']
        # = EV_R[n,j,:] · T_P[m,:]  → EV = np.einsum("njm,mm->njm", EV_R, T_P)
        # Let's be careful:
        # EV[n,j,m] = sum_m' T_P[m,m'] * EV_R[n,j,m']
        # In matrix notation per (n,j): row = T_P @ EV_R[n,j,:]   → shape [M]
        # Vectorised: EV[n,j,:] = T_P @ EV_R[n,j,:]
        EV = np.einsum("ml,njl->njm", T_P, EV_R)   # [N, J, M]

        # ── Bellman: Q[n, m, j] = π[n,m,j] + β * EV[n,j,m] ─────────────
        # pi_tensor: [N, M, J]   EV: [N, J, M] → need to align axes
        # Rearrange EV to [N, M, J]:
        EV_nmj = EV.transpose(0, 2, 1)  # [N, M, J]

        Q = pi_tensor + beta * EV_nmj   # [N, M, J]

        # ── Maximise over j ───────────────────────────────────────────────
        j_star  = np.argmax(Q, axis=2)          # [N, M]
        V_new   = Q[np.arange(N)[:, None],
                    np.arange(M)[None, :],
                    j_star]                     # [N, M]

        # ── Convergence check ─────────────────────────────────────────────
        diff = np.max(np.abs(V_new - V))
        V    = V_new

        if it % 50 == 0 or it == 1:
            elapsed = time.time() - t0
            print(f"  iter {it:4d}  |  max|ΔV| = {diff:.2e}  |  "
                  f"elapsed {elapsed:.1f}s", flush=True)

        if diff < tol:
            converged = True
            elapsed = time.time() - t0
            print(f"\n  Converged in {it} iterations  (max|ΔV|={diff:.2e})  "
                  f"elapsed {elapsed:.1f}s", flush=True)
            break

    if not converged:
        print(f"WARNING: VFI did not converge in {max_iter} iterations.", flush=True)

    # ── Extract optimal harvest policy ────────────────────────────────────────
    H_star = H_grid[j_star]   # [N, M]

    # ── Save results ──────────────────────────────────────────────────────────
    np.save("V_star.npy",  V)
    np.save("H_star.npy",  H_star)
    np.save("B_grid.npy",  B_grid)
    np.save("P_grid.npy",  P_grid)
    print("  Saved: V_star.npy, H_star.npy, B_grid.npy, P_grid.npy", flush=True)

    return V, H_star, B_grid, P_grid


# ─────────────────────────────────────────────────────────────────────────────
# Self-test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    p = PARAMS

    print("=" * 65)
    print("vfi.py — Value-Function Iteration for Iceland Cod")
    print("=" * 65)

    V_star, H_star, B_grid, P_grid = solve_vfi(p)

    # ── Total iterations already printed inside solve_vfi ────────────────────

    # ── Optimal harvest at (I0, P0) ──────────────────────────────────────────
    I0 = p["I0"]
    P0 = p["P0"]

    # Nearest grid indices
    n0 = int(np.argmin(np.abs(B_grid - I0)))
    m0 = int(np.argmin(np.abs(P_grid - P0)))

    H_opt = H_star[n0, m0]

    print(f"\n{'='*65}")
    print(f"  RESULTS")
    print(f"{'='*65}")
    print(f"  B_grid[n0] = {B_grid[n0]:>12,.0f} t  (closest to I0={I0:,.0f} t)")
    print(f"  P_grid[m0] = {P_grid[m0]:>12,.0f} ISK/t  (closest to P0={P0:,.0f})")
    print(f"  H*(I0,P0)  = {H_opt:>12,.0f} t")
    print(f"  ICES 2025 advice  = {p['catch_ices_2025']:>12,.0f} t")
    print(f"  Ratio H*/ICES     = {H_opt/p['catch_ices_2025']:.3f}")
    print(f"  V*(I0,P0)  = {V_star[n0,m0]:>20,.0f} ISK")
