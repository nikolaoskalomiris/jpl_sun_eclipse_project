#!/usr/bin/env python3
"""
diagnose_eclipse_geometry.py

Quick diagnostic for eclipse geometry and frame->time mapping.

Outputs geometry_diag.csv and prints whether any frame in the movie window
could possibly see any or all of the solar disk according to spherical geometry.

Edit USER CONFIG at the top to match your setup.
"""

import os, math
import numpy as np
import pandas as pd
import spiceypy as sp

# ---------------- USER CONFIG ----------------
KERNEL_DIR = r"C:\Users\SoR\Desktop\jpl_sun_eclipse_project\spice_kernels"

# The instant you think corresponds to the movie's TOTALITY_FRAME
TOTALITY_UTC_ISO = "2006-03-29 10:54:04.450"
TOTALITY_FRAME = 1000   # which frame (0-based) you assert corresponds to TOTALITY_UTC_ISO

FRAMES = 2000
FPS = 25.0

# Observer (same as used elsewhere)
OBSERVER_LAT_DEG = 36.14265853184001
OBSERVER_LON_DEG = 29.576375086997015
OBSERVER_ALT_M = 2.0

OUT_CSV = "geometry_diag.csv"
# ---------------- END USER CONFIG ----------------


def load_all_kernels(kernel_dir):
    if not os.path.isdir(kernel_dir):
        raise RuntimeError("KERNEL_DIR missing: " + kernel_dir)
    n = 0
    for fn in os.listdir(kernel_dir):
        if fn.lower().endswith((".bsp", ".tls", ".tpc", ".tf", ".tm")):
            try:
                sp.furnsh(os.path.join(kernel_dir, fn))
                n += 1
            except Exception:
                pass
    return n


def observer_j2000(et):
    try:
        _cnt, rvals = sp.bodvrd("EARTH", "RADII", 3)
        re_km = float(rvals[0]); rp_km = float(min(rvals))
    except Exception:
        re_km = 6378.137; rp_km = 6356.752
    lon_rad = math.radians(OBSERVER_LON_DEG)
    lat_rad = math.radians(OBSERVER_LAT_DEG)
    obs_body_km = sp.georec(lon_rad, lat_rad, OBSERVER_ALT_M / 1000.0, re_km, (re_km - rp_km) / re_km)
    frame_from = "ITRF93"
    try:
        xform = sp.pxform(frame_from, "J2000", et)
    except Exception:
        frame_from = "IAU_EARTH"
        xform = sp.pxform(frame_from, "J2000", et)
    obs_j2000 = sp.mxv(xform, obs_body_km)
    return np.array(obs_j2000, dtype=float)


def solar_angular_radius_rad_from_moon(sun_pos_wrt_moon):
    try:
        rvals = sp.bodvrd("SUN", "RADII", 3)[1]
        r_sun_km = float(rvals[0])
    except Exception:
        r_sun_km = 695700.0
    dist_km = float(np.linalg.norm(sun_pos_wrt_moon))
    if dist_km <= 0:
        return 0.0
    ratio = r_sun_km / dist_km
    ratio = max(-1.0, min(1.0, ratio))
    return math.asin(ratio)


def main():
    print("Loading kernels from:", KERNEL_DIR)
    loaded = load_all_kernels(KERNEL_DIR)
    print("Attempted to load {} kernel files.".format(loaded))

    et_total = sp.str2et(TOTALITY_UTC_ISO)
    start_et = et_total - (float(TOTALITY_FRAME) / float(FPS))
    print("Assuming frame 0 ET:", start_et, "(totality at frame {})".format(TOTALITY_FRAME))
    rows = []

    first_any_frame = None
    first_full_frame = None

    for frame in range(FRAMES):
        et = start_et + (frame / float(FPS))

        # moon -> observer (via moon position wrt Earth)
        moon_pos_wrt_earth, _ = sp.spkpos("MOON", et, "J2000", "NONE", "EARTH")
        moon_pos_wrt_earth = np.array(moon_pos_wrt_earth, dtype=float)

        # moon -> sun (direct vector)
        sun_pos_wrt_moon, _ = sp.spkpos("SUN", et, "J2000", "NONE", "MOON")
        sun_pos_wrt_moon = np.array(sun_pos_wrt_moon, dtype=float)

        obs_j2000 = observer_j2000(et)

        # u = moon -> observer
        u = obs_j2000 - moon_pos_wrt_earth
        un = np.linalg.norm(u)
        if un <= 0:
            print("Bad geometry at frame", frame); continue
        u_unit = u / un

        sd_norm = np.linalg.norm(sun_pos_wrt_moon)
        if sd_norm <= 0:
            print("Bad sun vector at frame", frame); continue
        sun_unit = sun_pos_wrt_moon / sd_norm

        # angle between u and sun (degrees)
        dot = float(np.dot(u_unit, sun_unit))
        dot = max(-1.0, min(1.0, dot))
        ang_deg = math.degrees(math.acos(dot))

        Rsun_rad = solar_angular_radius_rad_from_moon(sun_pos_wrt_moon)
        Rsun_deg = math.degrees(Rsun_rad)

        # thresholds
        threshold_any = 90.0 + Rsun_deg
        threshold_full = 90.0 - Rsun_deg

        any_possible = (ang_deg <= threshold_any)
        full_possible = (ang_deg <= threshold_full)

        rows.append((frame, sp.et2utc(et,"C",3), et, ang_deg, Rsun_deg, threshold_any, threshold_full, any_possible, full_possible))

        if first_any_frame is None and any_possible:
            first_any_frame = frame
        if first_full_frame is None and full_possible:
            first_full_frame = frame

        # print some diagnostics close to totality
        if abs(frame - TOTALITY_FRAME) <= 5 or frame % max(1, FRAMES // 20) == 0:
            print(f"frame {frame}: ang_between_deg={ang_deg:.6f} Rsun_deg={Rsun_deg:.6f} "
                  f"threshold_any={threshold_any:.6f} threshold_full={threshold_full:.6f} "
                  f"any_possible={any_possible} full_possible={full_possible}")

    df = pd.DataFrame(rows, columns=[
        "frame", "utc_iso", "et", "angle_sun_vs_u_deg", "solar_radius_deg",
        "threshold_any_deg", "threshold_full_deg", "any_possible", "full_possible"
    ])
    df.to_csv(OUT_CSV, index=False)
    print("Wrote", OUT_CSV)

    if first_any_frame is None:
        print("No frame in the window allows any part of solar disk to be visible (any_possible never True).")
    else:
        print("First frame allowing any part of Sun:", first_any_frame)

    if first_full_frame is None:
        print("No frame in the window allows the full disk to be visible (full_possible never True).")
    else:
        print("First frame allowing full disk:", first_full_frame)

    # quick summary stats
    min_ang = float(np.min(df['angle_sun_vs_u_deg'])); max_ang = float(np.max(df['angle_sun_vs_u_deg']))
    print(f"Angle range over frames: min={min_ang:.6f} deg max={max_ang:.6f} deg")
    print("Done.")

if __name__ == '__main__':
    main()
