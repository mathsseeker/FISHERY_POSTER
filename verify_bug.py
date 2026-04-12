"""
verify_bug.py — Confirm the w_s unit mismatch bug before any fix is applied.

Expected: population collapses within 20 years even at zero harvest,
because w_s is in tonnes/fish but x is in thousands of fish, making
dot(w_s, x) give biomass in (tonnes * thousands) instead of tonnes.
"""

import os
import numpy as np
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from params import PARAMS
from population import steady_state_x, transition

p = PARAMS

print("=" * 60)
print("verify_bug.py — Unit-mismatch diagnosis")
print("=" * 60)
print(f"\n  w_s (as loaded): {p['w_s']}  (units?)")
print(f"  w_s[0] = {p['w_s'][0]:.6f}  → {'tonnes/fish (BUG)' if p['w_s'][0] < 0.01 else 'kg/fish (OK)'}")

I0 = 972_145.0
x  = steady_state_x(I0, p)

print(f"\n  Initial x vector (from steady_state_x({I0:,.0f})):")
for s in range(4):
    print(f"    x[{s}] (age-{s+1}) = {x[s]:>18.1f}  "
          f"w*x = {p['w_s'][s]*x[s]:>14.1f} t")
print(f"  B0 = sum(w*x) = {np.dot(p['w_s'], x):>14,.1f} t")

print(f"\n  {'Year':>4}  {'B (t)':>14}  {'SSB (t)':>14}  {'x[0] recruits':>16}")
print("  " + "-" * 54)

from population import ssb

x_sim = x.copy()
B_history = []
for yr in range(1, 21):
    x_sim, B = transition(x_sim, H=0.0, eps_R=0.0, p=p)
    S = ssb(x_sim, p)
    B_history.append(B)
    print(f"  {yr:>4}  {B:>14,.1f}  {S:>14,.1f}  {x_sim[0]:>16.1f}")

B20 = B_history[-1]
print()
if B20 < 100_000:
    print(f"  BUG CONFIRMED: population collapses  (B year-20 = {B20:,.1f} t)")
elif B20 > 500_000:
    print(f"  NO BUG DETECTED  (B year-20 = {B20:,.1f} t)")
else:
    print(f"  AMBIGUOUS: B year-20 = {B20:,.1f} t  (between 100k and 500k)")
