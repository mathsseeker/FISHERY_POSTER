# FISHERY_POSTER — Iceland Cod Bioeconomic Model

Stochastic bioeconomic optimisation of the Icelandic cod fishery.
Solves for an optimal Harvest Control Rule by Value Function Iteration
on the 2D state `(biomass, price)`, calibrated to ICES 2025.

**Main finding: our price-aware policy generates ≈ 48 % more long-term
fishery value than Iceland's current ICES fixed rule, while
satisfying every ecological constraint.**

Poster project for Optimization II (IE, Year 2, Q2) — Katrín
Árnadóttir and Ricardo Pérez.

## Main results

At today's state `(B₀, P₀) = (1.07 Mt, 443 k ISK/t)`:

| Policy            | NPV (bn EUR) | Year-0 harvest (t) |
| ----------------- | ------------:| ------------------:|
| Optimal `H*(B,P)` | **6.07**     | **60 127**         |
| Fixed ICES rule   | 3.15         | 203 822            |

- Optimal policy prescribes **~70 % less harvest** than ICES advice
  today and still generates **~48 % more long-term value**
  (ΔV ≈ 2.92 bn EUR).
- 1 000 forward simulations × 20 years under the optimal policy →
  **collapse rate = 0 %**; stock rebuilds toward `B_MSY`.
- The fitted closure threshold `B_zero(P)` crosses the ICES management
  trigger `B_mgt = 220 kt` at today's price — an *emergent validation*
  of the ICES rule, which the model is not told about.

**Takeaway:** economic optimisation and sustainability are aligned,
not in conflict.

## Repository layout

```
.
├── run_all.py                        End-to-end pipeline
├── verify_bug.py / verify_fix.py     Bug / fix verification
├── src/                              Model modules
│                                       params, population, price,
│                                       profit, vfi, hcr, data loader
├── data/                             Input CSVs (ICES stock + ref points)
├── results/                          Computed .npy grids + policy
├── figures/                          hcr_plot.png, profit_check.png
└── docs/UNIT_ANALYSIS.md             Known kg/tonnes unit issue
```

## Running

```bash
python run_all.py
```

Run from the project root. Dependencies: `numpy`, `pandas`,
`matplotlib`, `scipy`.

## Known issues

See [`docs/UNIT_ANALYSIS.md`](docs/UNIT_ANALYSIS.md) — kg and tonnes
are mixed in a few places; reconcile before trusting the economic
magnitudes.

## Credits

Katrín Árnadóttir & Ricardo Pérez, 2026. Starting point: Pizarro &
Schwartz (2021), *Fisheries optimal harvest under price and biomass
uncertainty*. Stock data: ICES 2025 (`cod.27.5a`).
