#!/usr/bin/env python3
import sys, os
import pandas as pd
from PIL import Image, ImageDraw
import numpy as np

csv = sys.argv[1] if len(sys.argv)>1 else "merged_moon_limb_profile.csv"
out_dir = sys.argv[2] if len(sys.argv)>2 else "masks"
target_diameter = int(sys.argv[3]) if len(sys.argv)>3 else 2048  # pick your output canvas size

os.makedirs(out_dir, exist_ok=True)
df = pd.read_csv(csv)
frames = sorted(df['_frame_source'].unique())

# determine scale: your radius_px corresponds to some px in CSV; choose scale so that median radius -> target radius
median_radius_px = df['radius_px'].median()
target_radius = target_diameter//2
scale = target_radius / median_radius_px
cx, cy = target_radius, target_radius

for f in frames:
    sub = df[df['_frame_source']==f].sort_values('psi_deg')
    xs = (sub['x_px'].values * scale) + cx
    ys = (sub['y_px'].values * scale) + cy
    # build polygon points
    pts = list(zip(xs.tolist(), ys.tolist()))
    im = Image.new('L', (target_diameter, target_diameter), 255)  # white = visible by default
    draw = ImageDraw.Draw(im)
    # draw filled polygon of lunar limb -> that area will be occluded (black)
    draw.polygon(pts, fill=0)
    # optionally invert (so white=occluded)
    out_path = os.path.join(out_dir, f"mask_{int(f):05d}.png")
    im.save(out_path)
    if f % 50 == 0:
        print("wrote", out_path)
print("Done. Masks in:", out_dir)
