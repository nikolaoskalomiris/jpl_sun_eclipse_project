# sanitize_moon_limb_csv.py
# Usage: python sanitize_moon_limb_csv.py moon_limb_profile.csv

import sys
import pandas as pd
import numpy as np

CSV = sys.argv[1] if len(sys.argv) > 1 else "moon_limb_profile.csv"
OUT = CSV.replace(".csv", ".sanitized.csv")
GLD_MIN_LAT, GLD_MAX_LAT = -79.0, 79.0
GLD_MIN_LON, GLD_MAX_LON = -180.0, 180.0
MAX_RADIUS_PX = 1e5
MAX_ANG_RAD = np.pi

def read_csv(path):
    try:
        return pd.read_csv(path, low_memory=False)
    except Exception:
        return pd.read_csv(path, engine="python")

df = read_csv(CSV)
print("Read", len(df), "rows. Columns:", list(df.columns))

# ensure expected columns exist; if missing, try to infer indices
expected = ["psi_deg","lat_deg","lon_deg","elev_m","eff_radius_km","ang_rad","radius_px","x_px","y_px","sun_visible"]
for c in expected:
    if c not in df.columns:
        print("WARNING: column missing:", c)
        # create blank column so subsequent operations won't crash
        df[c] = np.nan

# fix lat/lon clamp
df["lat_deg"] = pd.to_numeric(df["lat_deg"], errors="coerce")
df["lon_deg"] = pd.to_numeric(df["lon_deg"], errors="coerce")
df["lat_deg_clamped"] = df["lat_deg"].clip(lower=GLD_MIN_LAT, upper=GLD_MAX_LAT)
df["lon_deg_wrapped"] = (((df["lon_deg"] + 180.0) % 360.0) - 180.0).clip(lower=GLD_MIN_LON, upper=GLD_MAX_LON)

# numeric sanitation
for c in ["elev_m","eff_radius_km","ang_rad","radius_px","x_px","y_px"]:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

# ang_rad clamp
df["ang_rad"] = df["ang_rad"].clip(lower=0.0, upper=MAX_ANG_RAD)

# radius clamp and fill NaNs with 0
df["radius_px"] = df["radius_px"].fillna(0.0)
df["radius_px"] = df["radius_px"].clip(lower=0.0, upper=MAX_RADIUS_PX)

# sun_visible to boolean
sv = df["sun_visible"].astype(str).str.lower().map({"true":True,"false":False,"1":True,"0":False,"t":True,"f":False})
df["sun_visible_bool"] = sv.fillna(False)

# choose to keep important columns and add flags
df_out = df.copy()
df_out["lat_deg"] = df["lat_deg_clamped"]
df_out["lon_deg"] = df["lon_deg_wrapped"]
df_out["sun_visible"] = df["sun_visible_bool"]

# drop helper cols
drop_cols = [c for c in df_out.columns if c.endswith("_clamped") or c.endswith("_wrapped") or c.endswith("_bool")]
# keep other original columns; but remove internal helpers for output
df_out = df_out.drop(columns=drop_cols, errors='ignore')

df_out.to_csv(OUT, index=False)
print("Wrote sanitized CSV to:", OUT)
print("Summary after sanitization:")
print(df_out[["lat_deg","lon_deg","ang_rad","radius_px","sun_visible"]].describe(include='all'))
