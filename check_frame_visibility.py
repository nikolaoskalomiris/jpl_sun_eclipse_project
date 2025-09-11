#!/usr/bin/env python3
"""
check_frame_visibility.py

Usage examples (Windows-friendly):
  python check_frame_visibility.py
  python check_frame_visibility.py --ae-csv eclipse_keyframes_full.csv --frames 990-1010:2
  python check_frame_visibility.py --ae-csv eclipse_keyframes_full.csv --frame-col frame_col --utc-col utc_iso --frames 995,1000,1005

This script:
 - Loads an AE CSV (eclipse_keyframes_full.csv or similar)
 - Detects the column holding AE frame numbers and the column holding UTC strings
 - Runs generate_moon_limb_profile_with_occlusion.py for a small set of frames (preview n-angles)
 - Reports sun-visible fraction per frame (quick sanity check)
"""
from __future__ import annotations
import argparse
import subprocess
import shlex
import os
import sys
import tempfile
import pandas as pd
from typing import List, Optional

DEFAULT_AE_CSV = "eclipse_keyframes_full.csv"
DEFAULT_HEAVY = "generate_moon_limb_profile_with_occlusion.py"
DEFAULT_KERNEL_DIR = "spice_kernels"
DEFAULT_DEM = "moon_dem/GLD100.tif"


def parse_frames_arg(s: Optional[str]) -> Optional[List[int]]:
    """Parse frames spec: '990-1010:2' or '990-1010' or '990,995,1000'"""
    if s is None:
        return None
    s = str(s).strip()
    out = []
    if "," in s:
        parts = [p.strip() for p in s.split(",") if p.strip() != ""]
        for p in parts:
            out.append(int(p))
        return out
    # range
    if "-" in s:
        step = 1
        if ":" in s:
            rng, step_s = s.split(":")
            step = int(step_s)
        else:
            rng = s
        a, b = [int(x) for x in rng.split("-", 1)]
        if b < a:
            a, b = b, a
        out = list(range(a, b + 1, step))
        return out
    # single frame
    return [int(s)]


def find_column(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    cols_lower = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in cols_lower:
            return cols_lower[c.lower()]
    # fallback: search for substring
    for c in df.columns:
        lc = c.lower()
        for cand in candidates:
            if cand.lower() in lc:
                return c
    return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ae-csv", default=DEFAULT_AE_CSV, help="AE CSV with frame/time mapping")
    p.add_argument("--heavy", default=DEFAULT_HEAVY, help="Path to heavy script to call (python script)")
    p.add_argument("--frames", default=None, help="Frames to test. Examples: '990-1010:2' or '995,1000' or '1000'")
    p.add_argument("--frame-col", default=None, help="Override frame column name in AE CSV")
    p.add_argument("--utc-col", default=None, help="Override UTC column name in AE CSV")
    p.add_argument("--use-row-index", action="store_true", help="Interpret provided values as CSV row indices instead of AE frames")
    p.add_argument("--preview-n-angles", type=int, default=256, help="Preview n-angles to pass to heavy script")
    p.add_argument("--no-multiproc", action="store_true", help="Pass --no-multiproc to heavy script (safer for debugging)")
    p.add_argument("--kernel-dir", default=DEFAULT_KERNEL_DIR)
    p.add_argument("--dem-path", default=DEFAULT_DEM)
    p.add_argument("--extra-heavy-args", default="", help="Extra args to add to heavy script call (quoted string)")
    args = p.parse_args()

    if not os.path.exists(args.ae_csv):
        print("AE CSV not found:", args.ae_csv)
        sys.exit(2)

    print("Loading AE CSV:", args.ae_csv)
    # try flexible CSV read (handles whitespace or comma-delimited)
    try:
        df = pd.read_csv(args.ae_csv, sep=None, engine="python")
    except Exception:
        df = pd.read_csv(args.ae_csv)

    print("AE CSV columns:", list(df.columns))
    # Determine frame column
    frame_col = args.frame_col
    if frame_col is None:
        frame_candidates = ["frame_col", "frame", "frame_index", "ae_frame", "csv_row", "row", "frame_num"]
        frame_col = find_column(df, frame_candidates)
    if frame_col is None:
        print("Could not detect a frame column automatically. Please specify --frame-col.")
        sys.exit(3)
    print("Using frame column:", frame_col)

    # Determine UTC column (ISO strings)
    utc_col = args.utc_col
    if utc_col is None:
        utc_candidates = ["utc_iso", "utc", "UTC", "time_iso", "time", "utc_time", "utc_iso_time"]
        utc_col = find_column(df, utc_candidates)
    if utc_col is None:
        # if time_s_center present we cannot easily reconstruct ISO here; so abort with instructions
        alt_candidates = ["time_s_center", "time_s", "et"]
        alt = find_column(df, alt_candidates)
        print("Could not find an ISO UTC column automatically.")
        if alt is not None:
            print(f"Found alternative column '{alt}'. If that column is seconds or ET please convert externally or provide --utc-col")
        print("Available columns:", list(df.columns))
        print("If your CSV uses a different column name for UTC (e.g. 'utc_iso'), pass --utc-col <name>.")
        sys.exit(4)
    print("Using UTC column:", utc_col)

    frames_list = parse_frames_arg(args.frames)
    if frames_list is None:
        # build a default test window centered in table order
        n = len(df)
        mid = n // 2
        # pick 21 tests every 4 rows by default
        rows = list(range(max(0, mid - 40), min(n, mid + 41), 4))
        print("No --frames given. Defaulting to CSV rows (by position) around middle of file.")
        # map those row indices to frame values depending on use-row-index
        if args.use_row_index:
            frames_to_test = rows
        else:
            # map csv row positions to frame_col values
            frames_to_test = []
            for r in rows:
                try:
                    fv = int(df.iloc[r][frame_col])
                except Exception:
                    fv = None
                if fv is not None:
                    frames_to_test.append(fv)
        print("Frames to test (auto):", frames_to_test[:10], " ... total:", len(frames_to_test))
    else:
        # frames provided by user are AE frames; if use-row-index True treat differently later
        frames_to_test = frames_list
        print("Frames to test (from --frames):", frames_to_test[:20])

    heavy = args.heavy
    if not os.path.exists(heavy):
        print("Heavy script not found:", heavy)
        sys.exit(5)

    # For each frame, find the corresponding CSV row (either by frame_col match or by row index)
    results = []
    for f in frames_to_test:
        if args.use_row_index:
            # f is csv row index into df
            if f < 0 or f >= len(df):
                print("Row index out of range:", f)
                continue
            utc = str(df.iloc[f][utc_col])
            frame_value = f
        else:
            # find row where frame_col == f (first match)
            matches = df[df[frame_col].astype(str) == str(f)]
            if matches.empty:
                print(f"Warning: frame {f} not found in column '{frame_col}'. Trying to match by integer equality...")
                try:
                    matches = df[df[frame_col].astype(float) == float(f)]
                except Exception:
                    matches = df[df.index == f]  # fallback to index
            if matches.empty:
                print("No matching row found for frame:", f, "Skipping.")
                continue
            row = matches.iloc[0]
            try:
                utc = str(row[utc_col])
            except Exception:
                print("Failed to extract UTC from row for frame:", f, "Row preview:", row.to_dict())
                continue
            frame_value = int(row[frame_col]) if str(row[frame_col]).isdigit() else f

        # Build heavy script command
        out_csv = f"tmp_vis_frame_{frame_value}.csv"
        cmd = []
        cmd.append(sys.executable)  # use same python interpreter
        cmd.append(heavy)
        cmd.append("--utc")
        cmd.append(utc)
        cmd.append("--out-csv")
        cmd.append(out_csv)
        cmd.append("--preview-n-angles")
        cmd.append(str(args.preview_n_angles))
        if args.no_multiproc:
            cmd.append("--no-multiproc")
        if args.kernel_dir:
            cmd.append("--kernel-dir")
            cmd.append(args.kernel_dir)
        if args.dem_path:
            cmd.append("--dem-path")
            cmd.append(args.dem_path)
        if args.extra_heavy_args:
            # split the extra args safely
            extra = shlex.split(args.extra_heavy_args)
            cmd.extend(extra)

        print("Running heavy script for frame", frame_value, "UTC:", utc)
        print("Command:", " ".join(shlex.quote(x) for x in cmd))
        rc = subprocess.run(cmd, shell=False)
        if rc.returncode != 0:
            print("Heavy script failed for frame", frame_value, "returncode", rc.returncode)
            continue
        if not os.path.exists(out_csv):
            print("Expected output not produced:", out_csv)
            continue
        try:
            rdf = pd.read_csv(out_csv)
            visible_frac = float(rdf['sun_visible'].sum()) / len(rdf)
        except Exception as e:
            print("Failed to read or parse heavy script output:", e)
            visible_frac = None
        results.append((frame_value, utc, visible_frac))
        # cleanup
        try:
            os.remove(out_csv)
        except Exception:
            pass

    print("\nResults (frame, utc, visible_frac):")
    for a, b, c in results:
        print(a, b, c)


if __name__ == "__main__":
    main()
