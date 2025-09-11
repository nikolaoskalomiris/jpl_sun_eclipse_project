#!/usr/bin/env python3
"""
make_ae_with_utc.py

Create a new AE CSV that includes an 'utc_iso' column constructed from
center_metadata.json (center_et or center_utc) + time_s_center column.

Usage:
  python make_ae_with_utc.py --ae-csv eclipse_keyframes_full.csv --center center_metadata.json --out eclipse_keyframes_full_with_utc.csv --kernel-dir spice_kernels
"""
import argparse, json, os, sys, glob
import pandas as pd
import spiceypy as sp

def load_spice_kernels(kernel_dir):
    loaded = []
    if not os.path.isdir(kernel_dir):
        return loaded
    # load all typical kernels
    for pat in ["*.tls", "*.tpc", "*.bsp", "*.tm", "*.tf", "*.tz"]:
        for fn in glob.glob(os.path.join(kernel_dir, pat)):
            try:
                sp.furnsh(fn)
                loaded.append(fn)
            except Exception:
                pass
    # load remaining files too
    for fn in glob.glob(os.path.join(kernel_dir, "*")):
        if fn not in loaded:
            try:
                sp.furnsh(fn)
                loaded.append(fn)
            except Exception:
                pass
    return loaded

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ae-csv", required=True)
    p.add_argument("--center", required=True, help="center_metadata.json (must contain center_et or center_utc)")
    p.add_argument("--out", default="eclipse_keyframes_full_with_utc.csv")
    p.add_argument("--kernel-dir", default="spice_kernels")
    p.add_argument("--time-col", default="time_s_center", help="column in AE CSV with seconds relative to center")
    args = p.parse_args()

    if not os.path.exists(args.ae_csv):
        print("AE CSV not found:", args.ae_csv); sys.exit(2)
    if not os.path.exists(args.center):
        print("Center metadata not found:", args.center); sys.exit(2)

    print("Loading AE CSV:", args.ae_csv)
    try:
        ae = pd.read_csv(args.ae_csv, sep=None, engine="python")
    except Exception:
        ae = pd.read_csv(args.ae_csv)

    print("Loading center metadata:", args.center)
    cm = json.load(open(args.center, "r"))

    center_et = None
    center_utc = None
    if "center_et" in cm:
        center_et = float(cm["center_et"])
        print("Using center_et from metadata:", center_et)
    elif "center_utc" in cm:
        center_utc = str(cm["center_utc"])
        print("Using center_utc from metadata:", center_utc)
    else:
        raise RuntimeError("center_metadata.json does not contain 'center_et' or 'center_utc'")

    print("Loading SPICE kernels (for et<->utc). Kernel dir:", args.kernel_dir)
    loaded = load_spice_kernels(args.kernel_dir)
    print("Kernels loaded:", len(loaded))

    if center_et is None:
        # convert center_utc to et
        center_et = sp.str2et(center_utc)
        print("Converted center_utc -> center_et:", center_et)

    if args.time_col not in ae.columns:
        raise RuntimeError(f"AE CSV does not contain expected time column '{args.time_col}'. Columns: {list(ae.columns)}")

    # compute per-row ET and utc_iso
    def make_utc_iso(delta_s):
        et = float(center_et) + float(delta_s)
        # use ISOC format (YYYY-MM-DDTHH:MM:SS.sss), convert T to space
        utc = sp.et2utc(et, "ISOC", 3)
        # replace T to space to match earlier files (optional)
        utc = utc.replace("T", " ")
        # strip trailing Z if present
        if utc.endswith("Z"):
            utc = utc[:-1]
        return utc

    print("Constructing utc_iso column from", args.time_col)
    ae["utc_iso"] = ae[args.time_col].apply(make_utc_iso)
    print("Preview (first 6 rows):")
    print(ae[["frame", args.time_col, "utc_iso"]].head(6).to_string(index=False))

    out_path = args.out
    ae.to_csv(out_path, index=False)
    print("Wrote:", out_path)

if __name__ == "__main__":
    main()
