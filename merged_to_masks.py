#!/usr/bin/env python3
"""
merged_to_masks.py

Convert a merged moon limb profile CSV (polar sampling per frame) into per-frame PNG masks.

Outputs per-frame:
 - mask_{frame:05d}.png        -> grayscale mask where 0 = occluded (moon), 255 = visible (background)
 - mask_{frame:05d}_alpha.png -> RGBA image with lunar silhouette opaque and background transparent

Usage examples:
  python merged_to_masks.py merged_moon_limb_profile.csv --out-dir masks --frame-col frame
  python merged_to_masks.py merged_moon_limb_profile.csv --out-dir masks --frame-col _frame_source --pad 8 --scale 2 --aa

What to run in cmd:
 python merged_to_masks.py merged_moon_limb_profile.csv --out-dir masks --frame-col frame --x-col x_px --y-col y_px --pad 8 --scale 2 --csv-out frame_mask_map.csv

Notes:
 - The script expects each row to contain x_px and y_px (Cartesian pixels, relative to some center) that describe the sampled limb contour.
 - If the merged CSV uses a different column name for frame index, pass --frame-col.
 - If x_px/y_px names differ, pass --x-col and --y-col.

Options:
  --pad N         : padding in pixels around silhouette (default 6)
  --scale S       : supersampling integer factor for anti-aliasing (default 1: no AA)
  --aa            : enable 2x internal supersample if scale==1 (convenience)
  --min-points M  : skip frames with fewer than M valid samples (default 8)
  --prefix P      : filename prefix (default "mask")
  --invert        : invert mask values (black->white)
  --frame-col     : column name with frame id (auto-detected)
  --x-col / --y-col: column names for x_px,y_px (auto-detected)
  --list-only     : print detected frames and extents without writing images
  --csv-out       : write a mapping CSV frames -> filepaths
"""

from __future__ import annotations
import argparse, os, sys, math
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

def find_column(df, candidates):
    cols_lower = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in cols_lower:
            return cols_lower[c.lower()]
    for c in df.columns:
        lc = c.lower()
        for cand in candidates:
            if cand.lower() in lc:
                return c
    return None

def polygon_from_points(xs, ys):
    return list(zip(xs.tolist(), ys.tolist()))

def main():
    p = argparse.ArgumentParser()
    p.add_argument("merged_csv")
    p.add_argument("--out-dir", default="masks")
    p.add_argument("--pad", type=int, default=6, help="padding in pixels around silhouette")
    p.add_argument("--scale", type=int, default=1, help="supersample scale factor for antialias (integer)")
    p.add_argument("--aa", action="store_true", help="enable simple 2x internal antialiasing if scale==1")
    p.add_argument("--min-points", type=int, default=8)
    p.add_argument("--prefix", default="mask")
    p.add_argument("--invert", action="store_true", help="invert mask output (white = occluded)")
    p.add_argument("--frame-col", default=None)
    p.add_argument("--x-col", default=None)
    p.add_argument("--y-col", default=None)
    p.add_argument("--list-only", action="store_true")
    p.add_argument("--csv-out", default="frame_mask_map.csv")
    args = p.parse_args()

    if args.aa and args.scale == 1:
        scale = 2
    else:
        scale = max(1, int(args.scale))

    if not os.path.exists(args.merged_csv := args.merged_csv):
        print("Missing input CSV:", args.merged_csv); sys.exit(2)

    df = pd.read_csv(args.merged_csv)
    print("Loaded CSV:", args.merged_csv, "rows:", len(df))
    # detect frame, x, y columns
    frame_col = args.frame_col
    if frame_col is None:
        frame_col = find_column(df, ["frame", "_frame_source", "frame_idx", "frame_index", "frame_col", "csv_row"])
    if frame_col is None:
        print("Failed to auto-detect frame column. Please pass --frame-col.")
        print("Available columns:", list(df.columns))
        sys.exit(3)
    print("Using frame column:", frame_col)

    x_col = args.x_col or find_column(df, ["x_px", "x", "x_px_proj", "x_pixel"])
    y_col = args.y_col or find_column(df, ["y_px", "y", "y_px_proj", "y_pixel"])
    if x_col is None or y_col is None:
        print("Failed to auto-detect x/y columns. Provide --x-col and --y-col. Columns available:", list(df.columns))
        sys.exit(4)
    print("Using x,y columns:", x_col, y_col)

    # ensure numeric
    df[x_col] = pd.to_numeric(df[x_col], errors="coerce")
    df[y_col] = pd.to_numeric(df[y_col], errors="coerce")
    df = df.dropna(subset=[x_col, y_col])
    # group by frame value
    groups = df.groupby(df[frame_col])
    frames = sorted(groups.groups.keys())
    print("Detected frames:", len(frames))

    if args.list_only:
        for f in frames:
            g = groups.get_group(f)
            xs = g[x_col].values
            ys = g[y_col].values
            print(f"frame {f}: pts={len(xs)} x_range=[{xs.min():.2f},{xs.max():.2f}] y_range=[{ys.min():.2f},{ys.max():.2f}]")
        return

    os.makedirs(args.out_dir, exist_ok=True)
    map_rows = []
    for f in frames:
        g = groups.get_group(f)
        xs = g[x_col].values.astype(float)
        ys = g[y_col].values.astype(float)
        if len(xs) < args.min_points:
            print(f"Skipping frame {f}: only {len(xs)} points (< {args.min_points})")
            continue
        # ensure samples are in angular order. If psi/angle exists, sort by it, else assume order is already contour
        if "psi_deg" in g.columns:
            order = np.argsort(g["psi_deg"].astype(float).values)
            xs = xs[order]
            ys = ys[order]
        # compute bounding box to allocate image
        min_x, max_x = float(xs.min()), float(xs.max())
        min_y, max_y = float(ys.min()), float(ys.max())
        # shift so that min_x/min_y plus pad are positive
        pad = float(args.pad)
        w = math.ceil((max_x - min_x) + 2*pad)
        h = math.ceil((max_y - min_y) + 2*pad)
        if w <= 0 or h <= 0:
            print(f"Invalid extents for frame {f}, skipping")
            continue

        # apply scale factor for supersampling (for anti-alias)
        sw = int(w * scale)
        sh = int(h * scale)

        # map points into image coords (0..w-1)
        xs_img = (xs - min_x + pad) * scale
        ys_img = (ys - min_y + pad) * scale
        # Note: image coords origin top-left (y increases downward). If your y_px is centered with positive up,
        # flip Y axis for correct orientation.
        # Heuristic: if mean(y_img) is roughly centered and values decrease/increase in expected direction, we keep as-is.
        # But many of your x_px,y_px appear to be centered with positive right and positive up -> flip Y:
        # Decide to flip Y so that positive y in input -> up in image -> corresponds to decreasing pixel row.
        # Implement flip: y' = height - 1 - y
        ys_img = (sh - 1) - ys_img

        polygon = polygon_from_points(xs_img, ys_img)

        # Create image and draw polygon (fill)
        mask = Image.new("L", (sw, sh), 255)  # default background 255 = visible
        draw = ImageDraw.Draw(mask)
        # convert polygon coords to sequence of tuples of ints
        poly_int = [(int(round(x)), int(round(y))) for x, y in polygon]
        # Ensure polygon is closed
        if poly_int[0] != poly_int[-1]:
            poly_int.append(poly_int[0])
        # Fill polygon with 0 (occluded)
        draw.polygon(poly_int, fill=0)

        # Downscale if supersampled
        if scale != 1:
            # Use ANTIALIAS/LANCZOS to downsample for smoothing edges
            mask = mask.resize((w, h), resample=Image.LANCZOS)

        # Optionally invert (if user wants 255=occluded)
        if args.invert:
            mask = Image.eval(mask, lambda px: 255 - px)

        # Save grayscale mask
        frame_num = int(float(f)) if isinstance(f, (int, float, np.integer, np.floating)) else str(f)
        fname_mask = os.path.join(args.out_dir, f"{args.prefix}_{int(frame_num):05d}.png")
        mask.save(fname_mask, compress_level=1)
        # Create alpha PNG: lunar silhouette opaque, background transparent
        rgba = Image.new("RGBA", mask.size, (0,0,0,0))
        # lunar silhouette where mask==0 -> opaque white or black? We'll set white with alpha 255.
        # Build alpha channel
        alpha = Image.eval(mask, lambda px: 255 - px) if not args.invert else mask.copy()
        # If args.invert False, mask has lunar=0 -> alpha = 255; else lunar=255 -> alpha accordingly.
        rgba.paste((255,255,255,255), (0,0), alpha)
        fname_alpha = os.path.join(args.out_dir, f"{args.prefix}_{int(frame_num):05d}_alpha.png")
        rgba.save(fname_alpha, compress_level=1)

        map_rows.append({"frame": frame_num, "mask": fname_mask, "alpha": fname_alpha, "w": w, "h": h})
        print(f"Written frame {frame_num} -> {fname_mask} ({w}x{h})")

    # write mapping CSV
    if args.csv_out:
        import csv
        with open(args.csv_out, "w", newline="") as fh:
            wcsv = csv.DictWriter(fh, fieldnames=["frame","mask","alpha","w","h"])
            wcsv.writeheader()
            for r in map_rows:
                wcsv.writerow(r)
        print("Wrote mapping CSV:", args.csv_out)

if __name__ == "__main__":
    main()
