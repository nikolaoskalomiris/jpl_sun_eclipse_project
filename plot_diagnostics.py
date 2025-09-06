#!/usr/bin/env python3
import sys
import pandas as pd
import matplotlib.pyplot as plt

path = sys.argv[1] if len(sys.argv)>1 else "merged_moon_limb_profile.csv"
df = pd.read_csv(path)
if '_frame_source' not in df.columns:
    print("Need _frame_source column (frame id).")
    sys.exit(1)

# normalize sun_visible to bool
sv = df['sun_visible'].astype(str).str.lower().map({'true':True,'false':False, '1':True,'0':False}).fillna(df['sun_visible']).astype(bool)
df['sun_visible_bool'] = sv

# compute fraction visible per frame
frac = df.groupby('_frame_source')['sun_visible_bool'].mean()
frac.index = frac.index.astype(int)
frac = frac.sort_index()

plt.figure(figsize=(10,3))
plt.plot(frac.index, frac.values, '-o', markersize=2)
plt.xlabel("frame")
plt.ylabel("fraction limb samples sun-visible")
plt.title("Sun-visible fraction per frame")
plt.grid(True)
plt.tight_layout()
plt.show()

# show median radius stability
rad_med = df.groupby('_frame_source')['radius_px'].median().sort_index()
plt.figure(figsize=(10,3))
plt.plot(rad_med.index, rad_med.values)
plt.xlabel("frame"); plt.ylabel("median radius_px"); plt.title("Radius stability")
plt.grid(True)
plt.show()
