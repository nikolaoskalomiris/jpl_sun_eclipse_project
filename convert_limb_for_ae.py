# convert_limb_for_ae.py
import math
import pandas as pd

# user settings (change to your comp / posMultiplier / anchor)
in_csv = "moon_limb_profile.csv"
out_csv = "moon_limb_profile_ae.csv"
posMultiplier = 16.488758     # your chosen scaling factor (comp px per sensor px)
comp_width = 4096.0
comp_height = 2160.0
anchor_x = comp_width / 2.0   # comp center
anchor_y = comp_height / 2.0  # comp center

df = pd.read_csv(in_csv)

# psi_deg is angle around limb; radius_px is in sensor pixels -> multiply by posMultiplier to get comp px
df['radius_comp_px'] = df['radius_px'] * posMultiplier
# AE coordinate system: +X right, +Y down (we invert Y from math)
df['ae_x'] = anchor_x + df['radius_comp_px'] * (df['psi_deg'].apply(math.radians).apply(math.cos))
df['ae_y'] = anchor_y - df['radius_comp_px'] * (df['psi_deg'].apply(math.radians).apply(math.sin))

# optional: only export psi samples where sun_visible == True (beads candidates)
df_beads = df[df['sun_visible'] == True].reset_index(drop=True)

# Save the full mapped file and a beads-only file
df.to_csv(out_csv, index=False)
df_beads.to_csv("moon_limb_profile_ae_beads.csv", index=False)

print("Wrote", out_csv, "and moon_limb_profile_ae_beads.csv")
print("Example rows:")
print(df[['psi_deg','ae_x','ae_y','radius_comp_px','sun_visible']].head())
