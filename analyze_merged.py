# analyze_merged.py
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("merged_moon_limb_profile.csv")  # path to your merged output
# If you have _frame_source or similar column, use it. Else adapt.
group = df.groupby("_frame_source") if "_frame_source" in df.columns else df.groupby("frame")
summary = group['sun_visible'].agg(['sum','count']).reset_index()
summary['visible_frac'] = summary['sum'] / summary['count']
summary.to_csv("frame_visibility_summary.csv", index=False)
print(summary.head())
# Quick plot
plt.plot(summary.iloc[:,0], summary['visible_frac'])
plt.xlabel("frame")
plt.ylabel("fraction sun-visible")
plt.title("Per-frame sun-visible fraction")
plt.grid(True)
plt.savefig("frame_visibility.png", dpi=150)
