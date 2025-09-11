#!/usr/bin/env python3
# spice_numba_debug.py
# Usage: python spice_numba_debug.py --kernel-dir spice_kernels --check-time "2006-03-29 10:54:04.555"

import os, sys, glob, argparse, traceback

def try_import(name):
    try:
        mod = __import__(name)
        return mod
    except Exception as e:
        return None

def furnsh_all(sp, kernel_dir):
    files = sorted(glob.glob(os.path.join(kernel_dir, "*")))
    cand = [f for f in files if os.path.isfile(f) and os.path.splitext(f)[1].lower() in (
        '.tls','.tpc','.bpc','.bsp','.tf','.tm','.txt','.tfs','.tfk')]
    ok, failed = [], []
    for p in cand:
        try:
            sp.furnsh(p)
            ok.append(p)
        except Exception as e:
            failed.append((p, str(e)))
    return ok, failed, cand

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--kernel-dir", default="spice_kernels")
    p.add_argument("--check-time", default="2006-03-29 10:54:04.555")
    args = p.parse_args()

    sp = try_import('spiceypy')
    if sp is None:
        print("[ERROR] spiceypy not importable. Install spiceypy into the same Python you run this with.")
        sys.exit(2)
    import spiceypy as sp

    print("spiceypy toolkit:", sp.tkvrsn("TOOLKIT"))
    kd = args.kernel_dir
    if not os.path.isdir(kd):
        print("[ERROR] kernel-dir not found:", kd); sys.exit(2)
    print("Kernel dir:", os.path.abspath(kd))

    ok, failed, cand = furnsh_all(sp, kd)
    print(f"Furnsh: succeeded {len(ok)} files, failed {len(failed)}")
    if ok:
        print(" First 40 furnished:")
        for f in ok[:40]:
            print("  ", os.path.basename(f))
    if failed:
        print(" Furnsh failures (sample):")
        for f, e in failed[:10]:
            print("  ", os.path.basename(f), "->", e)

    # show if candidate bpc or tf exist
    bpcs = [os.path.basename(x) for x in cand if x.lower().endswith('.bpc')]
    tfs = [os.path.basename(x) for x in cand if x.lower().endswith('.tf')]
    print("Binary PCK candidates (.bpc):", bpcs)
    print("Frame TF candidates (.tf):", tfs[:20])

    # try str2et
    tstr = args.check_time
    try:
        et = sp.str2et(tstr)
        print("str2et OK:", tstr, "-> ET =", et)
    except Exception as e:
        print("str2et FAILED:", e)
        et = None

    # pxform test
    def pxform_test(frm):
        if et is None:
            print("No ET available; skipping pxform test")
            return False
        try:
            m = sp.pxform(frm, "J2000", float(et))
            print(f"pxform OK: {frm} -> J2000; sample row:", m[0][:3])
            return True
        except Exception as e:
            print(f"pxform FAILED: {frm} -> J2000: {e}")
            traceback.print_exc(limit=1)
            return False

    print("\nTesting transforms...")
    ok_itrf = pxform_test("ITRF93")
    if not ok_itrf:
        print("ITRF93 transform not available. Try adding earth_latest_high_prec.bpc to spice_kernels/")
    else:
        print("ITRF93 is available.")

    # NUMBA / CUDA checks
    print("\nNUMBA / CUDA checks")
    numba = try_import('numba')
    if numba is None:
        print(" numba not importable in this interpreter.")
    else:
        import numba
        try:
            from numba import cuda
            print(" numba version:", numba.__version__)
            print(" cuda.is_available():", cuda.is_available())
            try:
                print(" cuda.gpus list:", list(cuda.gpus))
            except Exception as e:
                print(" cuda.gpus enumeration failed:", e)
            try:
                print(" Running cuda.detect() ...")
                cuda.detect()
            except Exception as e:
                print(" cuda.detect() raised:", e)
            if cuda.is_available():
                try:
                    a = cuda.device_array(10)
                    del a
                    print(" tiny device allocation succeeded")
                except Exception as e:
                    print(" tiny device allocation failed:", e)
        except Exception as e:
            print(" from numba import cuda failed:", e)
            traceback.print_exc()

    print("\nDone. If ITRF93 pxform failed, add a binary Earth PCK (earth_latest_high_prec.bpc) into spice_kernels/ and re-run this script.")

if __name__ == "__main__":
    main()
