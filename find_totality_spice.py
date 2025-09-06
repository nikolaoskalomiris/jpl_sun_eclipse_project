#!/usr/bin/env python3
"""
find_totality_spice.py

Finds the epoch (ET + UTC) of minimum apparent Sun-Moon separation as seen
from a ground observer using SPICE vectors (topocentric via observer offset).

Edit USER CONFIG (KERNEL_DIR, start_utc_iso, search_span_seconds) then run.
"""
import os, math
import numpy as np
import spiceypy as sp

# ---------------- USER CONFIG ----------------
KERNEL_DIR = r"C:\Users\SoR\Desktop\jpl_sun_eclipse_project\spice_kernels"
START_UTC_ISO = "2006-03-29 10:54:04.555"   # your initial guess (can be same as before)
SEARCH_HALF_SECONDS = 3 * 3600.0 / 2.0     # half-window to search (3 hours total -> half = 1.5h)
N_SAMPLES = 40001                           # coarse sampling (increase if you want)
OBSERVER_LAT_DEG = 36.14265853184001
OBSERVER_LON_DEG = 29.576375086997015
OBSERVER_ALT_M = 2.0
# ---------------- END USER CONFIG ----------------

def load_kernels(kernel_dir):
    if not os.path.isdir(kernel_dir):
        raise RuntimeError("KERNEL_DIR missing: " + kernel_dir)
    count = 0
    for fn in os.listdir(kernel_dir):
        if fn.lower().endswith((".bsp", ".tls", ".tpc", ".tf", ".tm")):
            try:
                sp.furnsh(os.path.join(kernel_dir, fn)); count += 1
            except Exception:
                pass
    return count

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
    return obs_j2000

def angle_between(a, b):
    da = np.array(a, dtype=float); db = np.array(b, dtype=float)
    na = np.linalg.norm(da); nb = np.linalg.norm(db)
    if na == 0 or nb == 0:
        return None
    cosv = float(np.dot(da, db) / (na*nb))
    cosv = max(-1.0, min(1.0, cosv))
    return math.degrees(math.acos(cosv))

def main():
    print("Loading kernels from:", KERNEL_DIR)
    loaded = load_kernels(KERNEL_DIR)
    print("Attempted to load {} kernel files.".format(loaded))

    et_center = sp.str2et(START_UTC_ISO)
    start_et = et_center - SEARCH_HALF_SECONDS
    end_et   = et_center + SEARCH_HALF_SECONDS
    print("Searching ET window (ET):", start_et, "->", end_et)

    times = np.linspace(start_et, end_et, N_SAMPLES)
    best_idx = None
    best_sep_deg = 1e9

    for i, et in enumerate(times):
        # vector from observer to Sun and to Moon (topocentric)
        obs_j2000 = observer_j2000(et)              # km (from Earth's center)
        sun_wrt_earth, _ = sp.spkpos("SUN", et, "J2000", "NONE", "EARTH")
        moon_wrt_earth, _ = sp.spkpos("MOON", et, "J2000", "NONE", "EARTH")
        sun_from_obs = np.array(sun_wrt_earth) - np.array(obs_j2000)
        moon_from_obs = np.array(moon_wrt_earth) - np.array(obs_j2000)
        sep = angle_between(sun_from_obs, moon_from_obs)
        if sep is not None and sep < best_sep_deg:
            best_sep_deg = sep; best_idx = i

    best_et = float(times[best_idx])
    best_utc = sp.et2utc(best_et, "C", 3)
    print("Found minimum separation at ET:", best_et, "UTC:", best_utc, "sep_deg:", best_sep_deg)
    # also print a small neighborhood for confirmation
    print("Neighborhood (frame-like samples around min):")
    for j in range(max(0,best_idx-5), min(len(times), best_idx+6)):
        et = float(times[j]); utc = sp.et2utc(et,"C",3)
        sun_wrt_earth, _ = sp.spkpos("SUN", et, "J2000", "NONE", "EARTH")
        moon_wrt_earth, _ = sp.spkpos("MOON", et, "J2000", "NONE", "EARTH")
        sep = angle_between(np.array(sun_wrt_earth) - observer_j2000(et), np.array(moon_wrt_earth) - observer_j2000(et))
        print("  idx", j, "UTC", utc, "sep_deg", sep)
    print("Done.")

if __name__ == "__main__":
    main()
