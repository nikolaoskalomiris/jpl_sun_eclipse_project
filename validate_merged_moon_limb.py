#!/usr/bin/env python3
import sys, os
import pandas as pd
import numpy as np

path = sys.argv[1] if len(sys.argv)>1 else "merged_moon_limb_profile.csv"
df = pd.read_csv(path)
print("Loaded:", path, "rows:", len(df))
print("Columns:", df.columns.tolist())

# Basic NaN checks
nan_counts = df.isna().sum()
print("\nNaN counts:\n", nan_counts[nan_counts>0])

# Expected fields
for col in ("psi_deg","lat_deg","lon_deg","elev_m","eff_radius_km","ang_rad","radius_px","x_px","y_px","sun_visible","_frame_source"):
    if col not in df.columns:
        print("Warning: missing column:", col)

# psi uniqueness per frame (if frame column present)
if '_frame_source' in df.columns:
    grouped = df.groupby('_frame_source')
    sample_counts = grouped.size()
    print("\nFrames:", sample_counts.size, "samples/frame min/median/max:", sample_counts.min(), sample_counts.median(), sample_counts.max())
else:
    print("No _frame_source column; can't report per-frame sample counts.")

# psi coverage and duplicates
psi_min,psi_max = df['psi_deg'].min(), df['psi_deg'].max()
print(f"\npsi_deg range: {psi_min} .. {psi_max} unique_count: {df['psi_deg'].nunique()}")
dups = df['psi_deg'].duplicated().sum()
print("psi duplicates in whole file:", dups)

# radius stats
print("\nradius_px stats:", df['radius_px'].min(), df['radius_px'].median(), df['radius_px'].max())

# sun_visible fraction
if 'sun_visible' in df.columns:
    sun_vis = df['sun_visible'].astype(str).str.lower().map({'true':True,'false':False, '1':True,'0':False}).fillna(df['sun_visible'])
    print("\nsun_visible values unique:", pd.unique(sun_vis)[:10])
    if '_frame_source' in df.columns:
        frac = df.assign(sun_bool=sun_vis).groupby('_frame_source')['sun_bool'].mean()
        print("sun_visible fraction per frame min/median/max:", frac.min(), frac.median(), frac.max())
        # print around center if present
        center_idx = int(len(frac)/2)
        print("sample around middle frames:\n", frac.iloc[max(0,center_idx-5):center_idx+6])
else:
    print("No sun_visible column.")

print("\nDone.")
