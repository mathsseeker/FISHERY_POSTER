# Unit Analysis: kg vs Tonnes Mixing

## Summary
Yes, there **is a unit mixing problem** in your model between **kg** (weight-at-age) and **tonnes** (biomass/harvest).

---

## Where the Mixing Occurs

### 1. **Weight-at-Age Definition (params.py)**
```python
"w_s": np.array([0.37, 0.56, 0.79, 2.04]),  # weight-at-age (kg) ICES 2025
```
- Explicitly stated as **kg** (from ICES 2025 Table 8)
- Age-1: 0.37 kg, Age-4+: 2.04 kg ✓

### 2. **Biomass Grid Definition (profit.py)**
```python
B_grid = np.linspace(0.01 * I_max, I_max, p["N"])  # I_max = 1,844,319 tonnes
```
- B_grid is in **tonnes**

### 3. **Age-Class Biomass Calculation (population.py)**
```python
def ssb(x):
    return float(np.dot(p["gamma_s"] * p["w_s"], x))  # w_s in kg!

def steady_state_x(B):  # B in tonnes
    raw = w * q * np.exp(-m)
    total = np.dot(w, raw)  # This already has kg units in it!
    return raw * (B / total)  # Dividing tonnes by units involving kg
```

**Problem:** If `w` is in kg, then `total = sum(w * raw)` has units of kg·(dimensionless) = **kg**. But we're dividing B (in tonnes) by kg to scale the abundance vector. This is dimensionally inconsistent.

### 4. **Catchability Check in VFI (vfi.py)**
```python
x_n = x_grid[n]  # From steady_state_x — mixed units!
catchable_n = float(np.dot(p["w_s"] * p["q_s"], x_n))  # Result in ???

H_j = H_grid[j]  # In tonnes
if H_j > catchable_n:  # Comparing tonnes to ???
    # infeasible
```

**Problem:** If w_s is kg and catchable is kg·(internal x units), but H_j is in tonnes, the comparison is invalid.

### 5. **Recruitment Output**
```python
"r1": 1.5136,  # Comment: "SSB in tonnes, recruits in thousands of fish"
"r2": 3.5569e-6,

x[0] = r1 * SSB * exp(-r2 * SSB)  # Output in thousands
```

**Problem:** Recruitment output is in thousands of fish, but stored in `x[0]`. Then:
```python
B_next = np.dot(w_s, x_next)  # (kg) × (thousands of fish) = ???
```

---

## What Should Happen

For dimensional consistency, one of these must be true:

### Option A: w_s Should be Unitless (Relative Weights)
- Reinterpret w_s = [0.37, 0.56, 0.79, 2.04] as **relative weight proportions**, not kg
- Then x would be in tonnes (biomass by age class)
- Computation: sum(relative_weight * biomass) = total_biomass ✓
- **But:** Contradicts the ICES table attribution

### Option B: w_s in Tonnes (Scaled Down)
- Convert w_s from kg to tonnes: w_s = [0.37, 0.56, 0.79, 2.04] / 1000
- w_s = [0.00037, 0.00056, 0.00079, 0.00204] tonnes
- Then if x is number of fish: (tonnes/fish) × (fish) = tonnes ✓
- Update params.py accordingly

### Option C: Missing Conversion in recruitment / x initialization
- Add factor of 1000 to account for recruitment units
- Or explicitly scale x values when initializing

---

## Recommended Fix

**Add a scaling factor comment and doc** in `steady_state_x()` and `transition()`:

```python
def steady_state_x(B: float, p: dict = PARAMS) -> np.ndarray:
    """
    Distribute biomass B across age classes.
    
    UNITS: Returns x where sum(w_s * x_s) = B (tonnes).
    This assumes w_s values are implicitly rescaled to match B units.
    """
```

**And check if a factor of 1/1000 should be applied** to either:
- w_s values in params.py (divide by 1000)
- Or r1/r2 recruitment parameters
- Or everywhere x is multiplied by w_s

---

## Files to Audit
1. `params.py` — check w_s units vs intended use
2. `population.py` — verify all w·x products have consistent units
3. `vfi.py` — ensure catchable_n and H_j are both in tonnes
4. `profit.py` — verify π = P·H - c1·H^c2 uses correct H units
