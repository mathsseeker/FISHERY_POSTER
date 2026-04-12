"""
Iceland Cod — Minimal Model-Ready Dataset
==========================================
Fetches annual cod catch from Statistics Iceland (Hagstofa) PX-Web API.
Merges with ICES (2025) Table 8 assessment data.
Output: iceland_cod_model_data.csv  (one row per year, 1955–2025)

Data sources:
  Hagstofa: https://px.hagstofa.is/pxen/pxweb/en/Atvinnuvegir/
               Atvinnuvegir__sjavarutvegur__aflatolur__afli_manudir/SJA01102.px/
  ICES 2025: cod.27.5a Table 6 (calendar year catches) and Table 8 (stock summary)

Run:
    python iceland_cod_data_1.py
"""

import json
import requests
import pandas as pd
import numpy as np
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
#  ICES TABLE 6 — Calendar year catches (hard-coded fallback)
#  Source: ICES (2025) cod.27.5a, Table 6 (calendar year column)
# ─────────────────────────────────────────────────────────────────────────────

ICES_TABLE6_CATCH = {
    1988: 377554, 1989: 363125, 1990: 335316, 1991: 307759,
    1992: 264834, 1993: 250704, 1994: 178138, 1995: 168592,
    1996: 180701, 1997: 203112, 1998: 243987, 1999: 260147,
    2000: 235092, 2001: 236702, 2002: 209544, 2003: 207246,
    2004: 228342, 2005: 213867, 2006: 197202, 2007: 171646,
    2008: 147676, 2009: 183320, 2010: 170025, 2011: 172218,
    2012: 196171, 2013: 223582, 2014: 222021, 2015: 230165,
    2016: 251219, 2017: 243945, 2018: 267221, 2019: 263025,
    2020: 270302, 2021: 265740, 2022: 242211, 2023: 217847,
    2024: 220336,
}

# ─────────────────────────────────────────────────────────────────────────────
#  FUNCTION 1 — Fetch annual cod catch from Hagstofa PX-Web API
# ─────────────────────────────────────────────────────────────────────────────

API_URL = (
    "https://px.hagstofa.is/pxen/api/v1/en"
    "/Atvinnuvegir/sjavarutvegur/aflatolur/afli_manudir/SJA01102.px"
)


def fetch_hagstofa_cod_catch() -> pd.DataFrame:
    """
    Fetch annual total cod catch from Statistics Iceland (Hagstofa) PX-Web API.

    Steps:
      1. GET metadata to discover variable codes.
      2. POST JSON query for cod, all vessels, all gear, all available years,
         annual total (month = "Years total").
      3. Parse response → DataFrame(year, catch_tonnes).
      4. Falls back to ICES_TABLE6_CATCH if API is unreachable.

    Returns
    -------
    pd.DataFrame  columns: year (int), catch_tonnes (float), catch_source (str)
    """
    print("[Hagstofa]  Fetching variable metadata ...")

    # ── Step 1: GET metadata ──────────────────────────────────────────────────
    try:
        r = requests.get(API_URL, timeout=20)
        r.raise_for_status()
        meta = r.json()
    except Exception as e:
        print(f"  WARNING: Hagstofa metadata request failed ({e}).")
        print("  Falling back to ICES Table 6 catch data.")
        return _ices_fallback_df()

    var_lookup = {v["code"]: v for v in meta.get("variables", [])}

    def find_code(var_code, search_text):
        """Return item code whose valueText matches search_text (case-insensitive)."""
        v = var_lookup.get(var_code, {})
        for code, text in zip(v.get("values", []), v.get("valueTexts", [])):
            if text.strip().lower() == search_text.strip().lower():
                return code
        # starts-with fallback
        for code, text in zip(v.get("values", []), v.get("valueTexts", [])):
            if text.strip().lower().startswith(search_text.strip().lower()):
                return code
        return None

    def find_var_containing(keyword):
        kw = keyword.lower()
        for vcode, vinfo in var_lookup.items():
            if any(kw in t.lower() for t in vinfo.get("valueTexts", [])):
                return vcode
        return None

    # Identify variable codes
    species_var = find_var_containing("cod")
    vessel_var  = find_var_containing("all vessels")
    gear_var    = find_var_containing("all fishing gear")
    month_var   = find_var_containing("years total")
    year_var    = next(
        (c for c, v in var_lookup.items()
         if all(x.isdigit() and len(x) == 4 for x in v.get("values", [""])[:3])),
        None,
    )

    if not all([species_var, vessel_var, gear_var, year_var, month_var]):
        print("  WARNING: Could not identify all Hagstofa variable codes.")
        print("  Falling back to ICES Table 6 catch data.")
        return _ices_fallback_df()

    cod_code        = find_code(species_var, "Cod")
    all_vessel_code = find_code(vessel_var,  "All vessels")
    all_gear_code   = find_code(gear_var,    "All fishing gear")
    annual_code     = find_code(month_var,   "Years total")
    available_years = var_lookup[year_var]["values"]

    if not all([cod_code, all_vessel_code, all_gear_code, annual_code]):
        print("  WARNING: Could not find required item codes.")
        return _ices_fallback_df()

    print(f"  Species={cod_code}, Vessel={all_vessel_code}, "
          f"Gear={all_gear_code}, Month={annual_code}")
    print(f"  Years available: {available_years[0]}–{available_years[-1]} "
          f"({len(available_years)} years)")

    # ── Step 2: POST data query ───────────────────────────────────────────────
    query = {
        "query": [
            {"code": species_var, "selection": {"filter": "item", "values": [cod_code]}},
            {"code": vessel_var,  "selection": {"filter": "item", "values": [all_vessel_code]}},
            {"code": gear_var,    "selection": {"filter": "item", "values": [all_gear_code]}},
            {"code": year_var,    "selection": {"filter": "item", "values": available_years}},
            {"code": month_var,   "selection": {"filter": "item", "values": [annual_code]}},
        ],
        "response": {"format": "json"},
    }

    try:
        r = requests.post(API_URL, json=query,
                          headers={"Content-Type": "application/json"},
                          timeout=20)
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        print(f"  WARNING: Hagstofa data query failed ({e}). Falling back.")
        return _ices_fallback_df()

    rows = data.get("data", [])
    if not rows:
        print("  WARNING: Hagstofa returned 0 rows. Falling back.")
        return _ices_fallback_df()

    # ── Step 3: Parse response ────────────────────────────────────────────────
    records = []
    for row in rows:
        key = row["key"]
        val = row["values"][0]
        year = int(key[3])
        catch_kg = float(val) if val not in (".", "..", "", "None", None) else None
        records.append({
            "year":         year,
            "catch_tonnes": round(catch_kg / 1000, 1) if catch_kg is not None else None,
            "catch_source": "Hagstofa",
        })

    df = (pd.DataFrame(records)
            .dropna(subset=["catch_tonnes"])
            .sort_values("year")
            .reset_index(drop=True))

    print(f"  Received {len(df)} annual totals from Hagstofa ({df['year'].min()}–{df['year'].max()}).")
    return df


def _ices_fallback_df() -> pd.DataFrame:
    """Build a DataFrame from the hard-coded ICES Table 6 catch data."""
    records = [
        {"year": y, "catch_tonnes": float(c), "catch_source": "ICES_Table6_fallback"}
        for y, c in ICES_TABLE6_CATCH.items()
    ]
    return pd.DataFrame(records).sort_values("year").reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
#  FUNCTION 2 — Validate Hagstofa vs ICES Table 6
# ─────────────────────────────────────────────────────────────────────────────

def validate_catch(df_hagstofa: pd.DataFrame, threshold_pct: float = 5.0):
    """
    Compare Hagstofa annual totals against ICES Table 6 catches.
    Flags any year where |Hagstofa - ICES| / ICES > threshold_pct %.

    Returns
    -------
    pd.DataFrame  with columns: year, hagstofa_t, ices_t, diff_pct, flag
    """
    rows = []
    for _, row in df_hagstofa.iterrows():
        y = int(row["year"])
        ices_c = ICES_TABLE6_CATCH.get(y)
        if ices_c is None:
            continue
        hag_c = row["catch_tonnes"]
        if hag_c is None:
            continue
        diff_pct = abs(hag_c - ices_c) / ices_c * 100
        rows.append({
            "year":        y,
            "hagstofa_t":  hag_c,
            "ices_t":      float(ices_c),
            "diff_pct":    round(diff_pct, 1),
            "flag":        "⚠️  DISCREPANCY" if diff_pct > threshold_pct else "ok",
        })
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
#  ICES TABLE 8 — Full stock summary (1955-2025), central estimates
#  Source: ICES (2025) cod.27.5a, Table 8
# ─────────────────────────────────────────────────────────────────────────────

ICES_TABLE8 = [
    # (year, SSB_t, B4plus_t, recruitment_000s, HR_central, catch_t)
    (1955, 764385, 2091599, 151043, 0.24, 545250),
    (1956, 615134, 1819295, 143786, 0.26, 486909),
    (1957, 592664, 1640587, 161490, 0.30, 455182),
    (1958, 705042, 1651179, 215085, 0.29, 517359),
    (1959, 655335, 1581060, 303576, 0.30, 459081),
    (1960, 603839, 1657275, 153884, 0.25, 470121),
    (1961, 477929, 1430278, 195920, 0.27, 377291),
    (1962, 515102, 1464176, 125467, 0.28, 388985),
    (1963, 471713, 1299082, 173339, 0.33, 408800),
    (1964, 431796, 1211340, 197640, 0.33, 437012),
    (1965, 331192, 1053223, 219524, 0.35, 387106),
    (1966, 301778, 1063541, 232947, 0.32, 353357),
    (1967, 283868, 1139640, 319675, 0.32, 335721),
    (1968, 251008, 1242034, 171396, 0.32, 381770),
    (1969, 357110, 1335708, 239662, 0.34, 403205),
    (1970, 359970, 1332743, 179779, 0.34, 475077),
    (1971, 259682, 1083850, 193102, 0.38, 444248),
    (1972, 235430,  979056, 142217, 0.39, 395166),
    (1973, 250251,  831287, 277802, 0.44, 369205),
    (1974, 192162,  909797, 187245, 0.40, 368133),
    (1975, 175460,  891560, 259209, 0.40, 364754),
    (1976, 146692,  948775, 368141, 0.36, 346253),
    (1977, 199363, 1296752, 144347, 0.26, 340086),
    (1978, 213897, 1307478, 224278, 0.27, 329602),
    (1979, 308772, 1409048, 237505, 0.29, 366462),
    (1980, 376046, 1511588, 141677, 0.30, 432237),
    (1981, 286233, 1254622, 145432, 0.33, 465032),
    (1982, 189352,  987995, 141178, 0.33, 380068),
    (1983, 147228,  803519, 227940, 0.36, 298049),
    (1984, 156488,  915367, 143675, 0.34, 282022),
    (1985, 169219,  942551, 140253, 0.37, 323428),
    (1986, 194145,  868728, 297676, 0.44, 364797),
    (1987, 145886,  987897, 249410, 0.39, 389915),
    (1988, 154603,  975352, 176506, 0.38, 377554),
    (1989, 158335,  947472,  97309, 0.36, 363125),
    (1990, 195330,  816470, 130351, 0.39, 335316),
    (1991, 158240,  699168, 113404, 0.40, 307759),
    (1992, 143948,  566250, 159696, 0.45, 264834),
    (1993, 111761,  585067, 129699, 0.35, 250704),
    (1994, 146693,  568031,  80850, 0.30, 178138),
    (1995, 168015,  565157, 142050, 0.31, 168592),
    (1996, 154855,  678953, 166092, 0.29, 180701),
    (1997, 189272,  796571,  92167, 0.29, 203112),
    (1998, 198943,  738557, 155963, 0.35, 243987),
    (1999, 175931,  728250,  76128, 0.33, 260147),
    (2000, 162751,  590537, 167720, 0.40, 235092),
    (2001, 158207,  664314, 155411, 0.33, 236702),
    (2002, 188352,  713929, 157981, 0.29, 209544),
    (2003, 186061,  742117, 180741, 0.30, 207246),
    (2004, 192781,  812040,  85022, 0.27, 228342),
    (2005, 223124,  729160, 154970, 0.28, 213867),
    (2006, 217113,  692836, 132338, 0.26, 197202),
    (2007, 203576,  668918,  95261, 0.23, 171646),
    (2008, 254696,  672940, 131329, 0.26, 147676),
    (2009, 236289,  747915, 117294, 0.23, 183320),
    (2010, 267988,  796854, 125885, 0.22, 170025),
    (2011, 327567,  843407, 165479, 0.22, 172218),
    (2012, 363158,  963525, 175676, 0.22, 196171),
    (2013, 388018, 1086421, 125208, 0.21, 223582),
    (2014, 355181, 1092927, 174466, 0.21, 222021),
    (2015, 461848, 1174463, 148520, 0.21, 230165),
    (2016, 405675, 1229426, 100183, 0.20, 251219),
    (2017, 535375, 1158603, 155029, 0.22, 243945),
    (2018, 526252, 1204517, 154886, 0.22, 267221),
    (2019, 472196, 1135211, 119665, 0.24, 263025),
    (2020, 418667, 1028124, 147693, 0.26, 270302),
    (2021, 423945, 1149657, 133342, 0.22, 265740),
    (2022, 442038, 1135770, 171319, 0.20, 242211),
    (2023, 425860, 1180543, 135982, 0.19, 217847),
    (2024, 398858, 1068566, 107434, 0.20, 220336),
    (2025, 386752,  972145, 117919,  None, None),  # ICES short-term forecast
]


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN — build and save the model dataset
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 65)
    print("Iceland Cod Data Builder — iceland_cod_model_data.csv")
    print("=" * 65)

    # ── Step 1: Hagstofa catch (with fallback) ────────────────────────────────
    df_hag = fetch_hagstofa_cod_catch()

    # ── Step 2: Validate against ICES Table 6 ────────────────────────────────
    if df_hag["catch_source"].iloc[0] == "Hagstofa":
        val_df = validate_catch(df_hag)
        flags  = val_df[val_df["flag"] != "ok"]
        if len(flags):
            print(f"\n  ⚠️  Hagstofa vs ICES Table 6 discrepancies (>{5}%):")
            print(flags[["year", "hagstofa_t", "ices_t", "diff_pct"]].to_string(index=False))
        else:
            print(f"  ✓  Hagstofa catch validated against ICES Table 6 — no discrepancies >5%.")

    # ── Step 3: Build ICES Table 8 DataFrame ─────────────────────────────────
    df_ices = pd.DataFrame(ICES_TABLE8, columns=[
        "year", "SSB_t", "B4plus_t", "recruitment_000s", "HR_ices", "catch_ices_t",
    ])

    # ── Step 4: Merge catch sources ───────────────────────────────────────────
    # Prefer Hagstofa catch where available; fill with ICES Table 6/8 otherwise.
    df = df_ices.merge(
        df_hag[["year", "catch_tonnes", "catch_source"]],
        on="year", how="left",
    )
    df["catch_t"] = df["catch_tonnes"].combine_first(df["catch_ices_t"])
    df["catch_source"] = df["catch_source"].fillna(
        df["catch_ices_t"].notna().map({True: "ICES_Table8", False: "missing"})
    )

    # ── Step 5: Derived variables ─────────────────────────────────────────────
    I_max = df["B4plus_t"].max()  # true historical peak (1955)

    df["I_t"]           = df["B4plus_t"]
    df["I_normalised"]  = (df["B4plus_t"] / I_max).round(4)
    df["log_B4plus"]    = np.log(df["B4plus_t"])
    df["delta_log_B"]   = df["log_B4plus"].diff().shift(-1).round(4)
    df["HR_data"]       = (df["catch_t"] / df["B4plus_t"]).round(4)
    df["recruits_t"]    = df["recruitment_000s"] * 1000
    df["R_over_SSB"]    = (df["recruits_t"].shift(-1) / df["SSB_t"]).round(6)
    df["stock_productivity"] = (df["catch_t"] / df["HR_data"].replace(0, np.nan)).round(0)

    # ── Step 6: Save ──────────────────────────────────────────────────────────
    KEEP = [
        "year", "catch_t", "SSB_t", "B4plus_t", "I_normalised",
        "recruitment_000s", "HR_ices", "HR_data",
        "delta_log_B", "R_over_SSB", "stock_productivity", "catch_source",
    ]
    df_out = df[KEEP].sort_values("year").reset_index(drop=True)
    df_out.to_csv("iceland_cod_model_data.csv", index=False)

    # Reference points
    REF = pd.DataFrame([
        ("B_lim",          125_000, "t",    "Bloss — biological viability floor (ICES 2025 Table 4)"),
        ("B_PA",           160_000, "t",    "Bpa = Blim × exp(1.645×0.15) (ICES 2016/Table 4)"),
        ("MSY_Btrigger",   265_000, "t",    "MSY Btrigger (ICES 2025 Table 4)"),
        ("HR_MSY",           0.22,  "rate", "HRMSY = proportion of B4+ (ICES 2025 Table 4)"),
        ("mgt_Btrigger",   220_000, "t",    "Iceland management plan SSB trigger (ICES Table 4)"),
        ("HR_mgt",           0.20,  "rate", "Iceland management plan HR (ICES Table 4)"),
        ("I_max",     int(I_max),   "t",    "True historical peak B4+ (1955, ICES 2025 Table 8)"),
    ], columns=["parameter", "value", "unit", "note"])
    REF.to_csv("iceland_cod_reference_points.csv", index=False)

    # ── Step 7: Print summary ─────────────────────────────────────────────────
    recent = df_out[df_out["year"] >= 2010]
    print(f"\n{'=' * 65}")
    print("DATASET SUMMARY (2010–2025)")
    print(f"{'=' * 65}")
    print(recent[["year", "catch_t", "B4plus_t", "HR_data", "delta_log_B"]].to_string(index=False))

    print(f"""
{'=' * 65}
KEY CALIBRATION NUMBERS
{'=' * 65}
I_max (true historical peak, 1955): {I_max:>12,.0f} t
B_lim (ICES 2025 Table 4)         : {125000:>12,} t
MSY Btrigger                       : {265000:>12,} t
HR_MSY                             : {0.22:>12.2f}
I0 2024 (confirmed, Table 8)       : {1068566:>12,} t
I0 2025 (forecast, Table 8)        : {972145:>12,} t

Mean annual catch 2010-2024        : {recent['catch_t'].dropna().mean():>12,.0f} t
Std  annual catch 2010-2024        : {recent['catch_t'].dropna().std():>12,.0f} t
Mean HR (data-derived)             : {recent['HR_data'].dropna().mean():>12.3f}
Mean Δlog(B4+)                     : {recent['delta_log_B'].dropna().mean():>12.3f}
Std  Δlog(B4+)                     : {recent['delta_log_B'].dropna().std():>12.3f}
{'=' * 65}
Files saved:
  iceland_cod_model_data.csv
  iceland_cod_reference_points.csv
""")


if __name__ == "__main__":
    main()
