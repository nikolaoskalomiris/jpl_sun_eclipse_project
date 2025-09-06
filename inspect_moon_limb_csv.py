# inspect_moon_limb_csv.py
# Usage: python inspect_moon_limb_csv.py moon_limb_profile.csv

import sys
import pandas as pd
import numpy as np

CSV = sys.argv[1] if len(sys.argv) > 1 else "moon_limb_profile.csv"
GLD_MIN_LAT, GLD_MAX_LAT = -79.0, 79.0
GLD_MIN_LON, GLD_MAX_LON = -180.0, 180.0

def try_read(path):
    # try normal read, then fall back to python engine if needed
    try:
        df = pd.read_csv(path, low_memory=False)
        return df
    except Exception as e:
        print("pandas.read_csv normal engine failed:", e)
    try:
        df = pd.read_csv(path, engine="python")
        return df
    except Exception as e:
        print("pandas.read_csv python engine failed:", e)
    raise RuntimeError("Failed to read CSV with pandas")

print("Reading:", CSV)
df = try_read(CSV)
print("Columns:", list(df.columns))
n = len(df)
print("Rows:", n)

# Expected columns (based on script)
expected = ["psi_deg","lat_deg","lon_deg","elev_m","eff_radius_km","ang_rad",
            "radius_px","x_px","y_px","sun_visible"]
missing = [c for c in expected if c not in df.columns]
if missing:
    print("WARNING: missing expected columns:", missing)
else:
    print("All expected columns present.")

# Quick dtype summary
print(df.dtypes)

# Numeric columns
numcols = [c for c in df.columns if df[c].dtype.kind in 'fiu']
print("Numeric columns:", numcols)

# Basic stats
print("\nNumeric summary (min, median, max, nan-count):")
for c in numcols:
    vals = df[c].replace([np.inf, -np.inf], np.nan)
    print(f"  {c:12s} min={vals.min():.6g} med={vals.median():.6g} max={vals.max():.6g} n_nan={vals.isna().sum()}")

# Important range checks
if "lat_deg" in df.columns:
    out_lat = df[(df["lat_deg"] < GLD_MIN_LAT) | (df["lat_deg"] > GLD_MAX_LAT)]
    print("lat out-of-range count:", len(out_lat))
    if len(out_lat) > 0:
        print("  sample out-of-range lat rows (first 5):")
        print(out_lat.head(5)[["psi_deg","lat_deg","lon_deg"]])

if "lon_deg" in df.columns:
    out_lon = df[(df["lon_deg"] < GLD_MIN_LON) | (df["lon_deg"] > GLD_MAX_LON)]
    print("lon out-of-range count:", len(out_lon))
    if len(out_lon) > 0:
        print(out_lon.head(5)[["psi_deg","lat_deg","lon_deg"]])

# radius sanity
if "radius_px" in df.columns:
    rp = df["radius_px"].replace([np.inf, -np.inf], np.nan)
    huge = rp[rp > 1e5]
    print("radius_px NaNs:", rp.isna().sum(), " >100k px count:", len(huge), " min/max:", rp.min(), rp.max())

# ang_rad sanity
if "ang_rad" in df.columns:
    ar = df["ang_rad"].replace([np.inf, -np.inf], np.nan)
    bad_ang = df[(ar < 0) | (ar > np.pi)]
    print("ang_rad out-of-[0,pi] count:", len(bad_ang))

# psi coverage
if "psi_deg" in df.columns:
    # basic coverage check
    ps = df["psi_deg"].dropna().astype(float)
    print("psi_deg min/max:", ps.min(), ps.max(), "unique count:", ps.nunique())
    duplicates = ps.duplicated().sum()
    print("psi_deg duplicates:", duplicates)
    # check if covers full circle
    if ps.min() < 0 or ps.max() > 360:
        print("psi_deg outside 0..360")

# sun_visible check
if "sun_visible" in df.columns:
    sv = df["sun_visible"]
    # accept True/False, 1/0, 'True','False'
    def to_bool_col(s):
        if s.dtype == bool:
            return s
        return s.astype(str).str.lower().map({"true":True,"false":False,"1":True,"0":False,"t":True,"f":False}).fillna(False)
    svb = to_bool_col(sv)
    print("sun_visible True count:", svb.sum(), "False count:", len(svb)-svb.sum())

# correlation checks
if set(["radius_px","sun_visible"]).issubset(df.columns):
    print("Avg radius_px when sun_visible True/False:")
    rp = df["radius_px"].replace([np.inf, -np.inf], np.nan)
    for val in [True, False]:
        mask = (to_bool_col(df["sun_visible"]) == val)
        subset = rp[mask]
        print(f"  {val}: n={len(subset)} mean={subset.mean():.6g} median={subset.median():.6g}")

print("\nDone. If you want, run the sanitiser to automatically clamp/clean problematic fields.")
