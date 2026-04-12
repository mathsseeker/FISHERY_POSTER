"""
verify_fix.py — Confirm the w_s unit fix resolves the population collapse.

Simulation A: zero harvest, zero shock, 20 years → B year-20 should be > 500,000 t
Simulation B: HR=0.21 harvest, zero shock, 10 years → H year-1 should be 150k–280k t
"""

import os
import numpy as np
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from params import PARAMS
from population import steady_state_x, transition, ssb

p = PARAMS

print("=" * 65)
print("verify_fix.py — Unit-fix validation")
print("=" * 65)
print(f"\n  w_s[0] = {p['w_s'][0]:.5f}  "
      f"({'kg/fish ✓' if p['w_s'][0] > 0.1 else 'tonnes/fish — BUG STILL PRESENT'})")

I0 = 972_145.0

# ── Simulation A: zero harvest, zero shock ────────────────────────────────────
print("\n" + "─" * 65)
print("SIMULATION A — zero harvest, zero shock, 20 years")
print("─" * 65)
print(f"  {'Year':>4}  {'B (t)':>14}  {'SSB (t)':>14}  {'Recruits (000s)':>16}")
print("  " + "-" * 55)

x = steady_state_x(I0, p)
B_A = []
for yr in range(1, 21):
    x, B = transition(x, H=0.0, eps_R=0.0, p=p)
    S    = ssb(x, p)
    B_A.append(B)
    print(f"  {yr:>4}  {B:>14,.1f}  {S:>14,.1f}  {x[0]:>16.1f}")

B20_A = B_A[-1]
units_ok = B20_A > 500_000

# ── Simulation B: HR=0.21 harvest ────────────────────────────────────────────
print("\n" + "─" * 65)
print("SIMULATION B — HR=0.21 harvest, zero shock, 10 years")
print("─" * 65)
print(f"  {'Year':>4}  {'B_start (t)':>14}  {'H (t)':>12}  {'B_end (t)':>14}")
print("  " + "-" * 50)

x = steady_state_x(I0, p)
H_year1 = None
for yr in range(1, 11):
    B_start     = float(np.dot(p["w_s"], x))
    catchable   = float(np.dot(p["w_s"] * p["q_s"], x))
    H           = min(0.21 * B_start, catchable)
    if yr == 1:
        H_year1 = H
    x, B_end = transition(x, H=H, eps_R=0.0, p=p)
    print(f"  {yr:>4}  {B_start:>14,.1f}  {H:>12,.1f}  {B_end:>14,.1f}")

h_plausible = 150_000 <= H_year1 <= 280_000

# ── Final verdict ─────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("CHECKS")
print("=" * 65)
print(f"  B year-20 (Sim A): {B20_A:>12,.1f} t  "
      f"  [threshold > 500,000]  {'✓ PASS' if units_ok else '✗ FAIL'}")
print(f"  H year-1  (Sim B): {H_year1:>12,.1f} t  "
      f"  [threshold 150k–280k]  {'✓ PASS' if h_plausible else '✗ FAIL'}")
print()

if units_ok:
    print("  UNITS OK")
if h_plausible:
    print("  H* PLAUSIBLE")
if units_ok and h_plausible:
    print("  ALL CHECKS PASSED ✓")
else:
    print("  ONE OR MORE CHECKS FAILED — do not proceed to Task 5")
    print(f"    B20={B20_A:,.0f}  H1={H_year1:,.0f}")
