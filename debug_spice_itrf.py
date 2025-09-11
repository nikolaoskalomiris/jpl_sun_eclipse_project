#!/usr/bin/env python3
# debug_spice_itrf.py
# Usage:
#   python debug_spice_itrf.py [kernel_dir] [optional UTC e.g. "2006-03-29 10:54:04.555"]
#
# Place this script next to your spice_kernels folder or pass the full path.

import os
import sys
import traceback
import spiceypy as sp

def furnsh_all_in_dir(kdir):
    print(f"Kernel dir: {kdir}")
    if not os.path.isdir(kdir):
        print("ERROR: kernel dir not found:", kdir)
        return []
    files = sorted(os.listdir(kdir))
    furnished = []
    errors = []
    for fn in files:
        p = os.path.join(kdir, fn)
        # only try plausible kernel types
        if not any(fn.lower().endswith(ext) for ext in (".tls", ".tpc", ".bsp", ".tf", ".tm", ".tfs", ".tfk", ".txt")):
            continue
        try:
            sp.furnsh(p)
            furnished.append(p)
        except Exception as e:
            errors.append((p, str(e)))
    print(f"\nFurnished {len(furnished)} kernel files (first 40 shown):")
    for p in furnished[:40]:
        print("  ", p)
    if errors:
        print(f"\nFailed to furnsh {len(errors)} files:")
        for p,e in errors:
            print("  ", p, " -> ", e)
    return furnished

def try_pxform_pairs(et):
    pairs = [
        ("ITRF93", "IAU_EARTH"),
        ("ITRF93", "J2000"),
        ("IAU_EARTH", "J2000"),
        ("ITRF93", "ITRF93"),
        ("ITRF93", "ITRF2000"),
    ]
    print("\nAttempting pxform conversions for ET =", et)
    for a,b in pairs:
        try:
            mat = sp.pxform(a, b, float(et))
            print(f"pxform OK: {a} -> {b}. Matrix (first row): {mat[:3]}")
        except Exception as e:
            print(f"pxform FAIL: {a} -> {b}. Exception:")
            traceback.print_exception(type(e), e, e.__traceback__)

def main():
    # parse args
    kernel_dir = "spice_kernels"
    utc_input = None
    if len(sys.argv) >= 2:
        kernel_dir = sys.argv[1]
    if len(sys.argv) >= 3:
        utc_input = sys.argv[2]

    print("=== debug_spice_itrf.py ===")
    print("spiceypy version:", sp.tkvrsn("TOOLKIT"))

    # Step 1: furnsh all kernels found in the folder
    furnished = furnsh_all_in_dir(kernel_dir)

    # Force-show whether a file called 'earth_assoc_itrf93.tf' exists
    candidate = os.path.join(kernel_dir, "earth_assoc_itrf93.tf")
    print("\nearth_assoc_itrf93.tf present?", os.path.exists(candidate))
    if os.path.exists(candidate):
        print("  -> size:", os.path.getsize(candidate))

    # Step 2: choose an ET to test
    if utc_input is None:
        # try to look for center_metadata.json in cwd as in your pipeline
        if os.path.exists("center_metadata.json"):
            import json
            try:
                cm = json.load(open("center_metadata.json", "r"))
                center_et = cm.get("center_et", None)
                center_utc = cm.get("center_utc", None)
                if center_et:
                    print("\nUsing center_et from center_metadata.json:", center_et)
                    et = float(center_et)
                elif center_utc:
                    print("\nUsing center_utc from center_metadata.json:", center_utc)
                    et = sp.str2et(center_utc)
                else:
                    et = sp.str2et("2006-03-29 10:54:04.555")
            except Exception as e:
                print("Could not read center_metadata.json:", e)
                et = sp.str2et("2006-03-29 10:54:04.555")
        else:
            et = sp.str2et("2006-03-29 10:54:04.555")
    else:
        try:
            et = float(utc_input)
            print("Using ET numeric from argv:", et)
        except Exception:
            try:
                et = sp.str2et(utc_input)
            except Exception as e:
                print("str2et failed for provided UTC:", e)
                et = sp.str2et("2006-03-29 10:54:04.555")

    print("\nET chosen:", et)

    # small check: try to call pxform with ITRF93 now
    try_pxform_pairs(et)

    # Additional diagnostic: list frames known? we can't list all frames easily, but we can attempt to check for some known frame ids
    print("\nAdditional diagnostics:")
    test_frames = ["ITRF93", "ITRF2000", "IAU_EARTH", "J2000"]
    for f in test_frames:
        try:
            # pxform to itself to test presence
            sp.pxform(f, f, float(et))
            print(f"Frame {f} seems present (pxform to itself succeeded).")
        except Exception as e:
            print(f"Frame {f} not usable: {e}")

    # show kernel pool summary (limited)
    try:
        pool_count = sp.kdata(0)[0]  # kdata can be used to query kernel pool but this usage is not always trivial
        print("\nsp.kdata returned (sample):", pool_count)
    except Exception:
        # ignore; not critical
        pass

    print("\nDone. If pxform(ITRF93,...) still fails, paste this entire output and the list of furnished kernel files here.")
    print("Hint: ensure a frame kernel (FK/TF) that defines ITRF93 is furnished. Typical name: earth_assoc_itrf93.tf")

if __name__ == "__main__":
    main()
