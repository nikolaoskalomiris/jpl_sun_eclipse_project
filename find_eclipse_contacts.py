#!/usr/bin/env python3
"""
find_eclipse_contacts.py

Finds approximate first/second/third/fourth contact times for a solar eclipse
as seen from a given ground observer, using SPICE.

How it works:
 - computes apparent angular radii of Sun and Moon as seen from the observer
 - computes center-to-center angular separation (Sun vs Moon) as seen from observer
 - finds times where separation == R_sun + R_moon (first/fourth contact)
   and separation == abs(R_moon - R_sun) (second/third contact)
 - uses coarse scan + binary-refinement for each crossing.

Edit USER CONFIG block below to match your kernel dir / observer / initial guess.
Run and use the printed UTC times to set TOTALITY_UTC_ISO or pick frames.
"""

import os, math, sys
import numpy as np
import spiceypy as sp
import pandas as pd
from datetime import timedelta

# ---------------- USER CONFIG ----------------
KERNEL_DIR = r"C:\Users\SoR\Desktop\jpl_sun_eclipse_project\spice_kernels"
# initial guess (center) you already found:
CENTER_UTC_ISO = "2006-03-29 10:54:04.555"

# how far to search outward (seconds) from center for contacts
SEARCH_HALF_WINDOW_S = 6 * 3600.0   # search ±6 hours (safe)
# coarse sampling spacing (seconds) for initial scanning
COARSE_STEP_S = 1.0                 # 1 second sampling; increase for speed, decrease for accuracy
# binary refinement tolerance (seconds)
REFINE_TOL_S = 1e-3  # sub-millisecond is overkill; set to 1e-3s for precision

FPS = 25.0
TOTALITY_FRAME = 1000  # which frame corresponds to CENTER_UTC_ISO in your mapping

OUT_CSV = "contacts.csv"
# ---------------- END USER CONFIG ----------------

def load_kernels(kernel_dir):
    if not os.path.isdir(kernel_dir):
        raise RuntimeError("KERNEL_DIR not found: " + kernel_dir)
    count = 0
    for fn in os.listdir(kernel_dir):
        if fn.lower().endswith((".bsp", ".tls", ".tpc", ".tf", ".tm")):
            try:
                sp.furnsh(os.path.join(kernel_dir, fn))
                count += 1
            except Exception:
                pass
    return count

def observer_j2000(et, lat_deg=None, lon_deg=None, alt_m=None):
    # uses the globals for lat/lon/alt if not given
    if lat_deg is None:
        lat_deg = 36.14265853184001
    if lon_deg is None:
        lon_deg = 29.576375086997015
    if alt_m is None:
        alt_m = 2.0
    try:
        _cnt, rvals = sp.bodvrd("EARTH", "RADII", 3)
        re_km = float(rvals[0]); rp_km = float(min(rvals))
    except Exception:
        re_km = 6378.137; rp_km = 6356.752
    lon_rad = math.radians(lon_deg)
    lat_rad = math.radians(lat_deg)
    obs_body_km = sp.georec(lon_rad, lat_rad, alt_m / 1000.0, re_km, (re_km - rp_km) / re_km)
    frame_from = "ITRF93"
    try:
        xform = sp.pxform(frame_from, "J2000", et)
    except Exception:
        frame_from = "IAU_EARTH"
        xform = sp.pxform(frame_from, "J2000", et)
    obs_j2000 = sp.mxv(xform, obs_body_km)
    return np.array(obs_j2000, dtype=float)

def apparent_angular_radius_km(body, dist_km):
    """
    Returns angular radius (radians) of body with NAIF body radii or fallback values.
    body: 'SUN' or 'MOON' or other
    dist_km: distance observer->body center (km)
    """
    try:
        _cnt, rvals = sp.bodvrd(body, "RADII", 3)
        r_km = float(rvals[0])
    except Exception:
        if body.upper() == "SUN":
            r_km = 695700.0
        elif body.upper() == "MOON":
            # mean moon radius ~1737.4 km
            r_km = 1737.4
        else:
            raise
    if dist_km <= 0:
        return 0.0
    ratio = r_km / dist_km
    ratio = max(-1.0, min(1.0, ratio))
    return math.asin(ratio)

def sep_deg_at_et(et):
    """
    Returns (sep_deg, Rsun_deg, Rmoon_deg) for the given epoch ET.
    sep_deg = angular separation between Sun and Moon centers as seen from observer (deg).
    Rsun_deg, Rmoon_deg: apparent angular radii in degrees.
    """
    # moon & sun positions wrt Earth and wrt Moon as needed
    moon_wrt_earth, _ = sp.spkpos("MOON", et, "J2000", "NONE", "EARTH")
    sun_wrt_earth, _ = sp.spkpos("SUN", et, "J2000", "NONE", "EARTH")

    obs_j2000 = observer_j2000(et)
    # vectors from observer to body centers
    v_moon = np.array(moon_wrt_earth, dtype=float) - np.array(obs_j2000, dtype=float)
    v_sun  = np.array(sun_wrt_earth, dtype=float) - np.array(obs_j2000, dtype=float)

    d_moon = np.linalg.norm(v_moon)
    d_sun  = np.linalg.norm(v_sun)
    if d_moon <= 0 or d_sun <= 0:
        return None

    # center-to-center separation
    cosc = float(np.dot(v_moon, v_sun) / (d_moon * d_sun))
    cosc = max(-1.0, min(1.0, cosc))
    sep_deg = math.degrees(math.acos(cosc))

    Rsun_rad = apparent_angular_radius_km("SUN", d_sun)
    Rmoon_rad = apparent_angular_radius_km("MOON", d_moon)

    return sep_deg, math.degrees(Rsun_rad), math.degrees(Rmoon_rad)

def find_crossing(center_et, target_value_deg, search_dir=1, coarse_step_s=1.0, half_window_s=3600.0, tol_s=1e-3):
    """
    Find first epoch where sep_deg crosses target_value_deg moving outward from center_et in direction search_dir (+1 forward, -1 backward).
    Strategy: coarse stepping until sign change then refine with binary search.
    Returns epoch ET (float) or None if not found within half_window_s.
    """
    max_steps = int(math.ceil(half_window_s / coarse_step_s))
    et0 = center_et
    prev_et = et0
    prev_sep, _, _ = sep_deg_at_et(prev_et)
    # take steps outward
    for i in range(1, max_steps+1):
        et = center_et + search_dir * i * coarse_step_s
        s, _, _ = sep_deg_at_et(et)
        # check if we crossed target (we want s - target to change sign)
        if (prev_sep - target_value_deg) * (s - target_value_deg) <= 0:
            # bracket found between prev_et and et -> refine
            a = prev_et; b = et
            # ensure a < b
            if a > b:
                a, b = b, a
            # binary refine
            fa = sep_deg_at_et(a)[0] - target_value_deg
            fb = sep_deg_at_et(b)[0] - target_value_deg
            if fa == 0.0:
                return a
            if fb == 0.0:
                return b
            # refine loop
            while (b - a) > tol_s:
                m = 0.5 * (a + b)
                fm = sep_deg_at_et(m)[0] - target_value_deg
                if fa * fm <= 0:
                    b = m; fb = fm
                else:
                    a = m; fa = fm
            return 0.5*(a+b)
        prev_et = et; prev_sep = s
    return None

def main():
    print("Loading SPICE kernels from:", KERNEL_DIR)
    n = load_kernels(KERNEL_DIR)
    print("Attempted to load {} kernel files.".format(n))

    center_et = sp.str2et(CENTER_UTC_ISO)
    print("Center UTC ISO:", CENTER_UTC_ISO, "ET:", center_et)

    # compute base sep / radii at center
    sep_deg, Rsun_deg, Rmoon_deg = sep_deg_at_et(center_et)
    print("At center: sep_deg =", sep_deg, "R_sun_deg=", Rsun_deg, "R_moon_deg=", Rmoon_deg)

    # thresholds for contacts (observer-centric)
    thr_1_4 = Rsun_deg + Rmoon_deg        # first / fourth contact threshold (outer)
    thr_2_3 = abs(Rmoon_deg - Rsun_deg)   # second / third contact threshold (inner)

    print("Contact thresholds (deg): first/fourth =", thr_1_4, " second/third =", thr_2_3)

    # find times: search outward from center for 1st (backwards) and 4th (forwards) where sep crosses thr_1_4
    print("Searching for first contact (sep == R_sun + R_moon) backward from center ...")
    et_first = find_crossing(center_et, thr_1_4, search_dir=-1, coarse_step_s=COARSE_STEP_S, half_window_s=SEARCH_HALF_WINDOW_S, tol_s=REFINE_TOL_S)
    print("Searching for fourth contact (sep == R_sun + R_moon) forward from center ...")
    et_fourth = find_crossing(center_et, thr_1_4, search_dir=+1, coarse_step_s=COARSE_STEP_S, half_window_s=SEARCH_HALF_WINDOW_S, tol_s=REFINE_TOL_S)

    print("Searching for second contact (sep == |R_moon - R_sun|) forward from center ...")
    et_second = find_crossing(center_et, thr_2_3, search_dir=+1, coarse_step_s=COARSE_STEP_S, half_window_s=SEARCH_HALF_WINDOW_S, tol_s=REFINE_TOL_S)
    print("Searching for third contact (sep == |R_moon - R_sun|) backward from center ...")
    et_third = find_crossing(center_et, thr_2_3, search_dir=-1, coarse_step_s=COARSE_STEP_S, half_window_s=SEARCH_HALF_WINDOW_S, tol_s=REFINE_TOL_S)

    rows = []
    def add_row(name, et):
        if et is None:
            rows.append((name, None, None, None, None))
            return
        sep, Rs, Rm = sep_deg_at_et(et)
        rows.append((name, sp.et2utc(et,"C",3), et, sep, Rs, Rm))

    add_row("first_contact", et_first)
    add_row("second_contact", et_second)
    add_row("center", center_et)
    add_row("third_contact", et_third)
    add_row("fourth_contact", et_fourth)

    df = pd.DataFrame(rows, columns=["contact","utc_iso","et","sep_deg","Rsun_deg","Rmoon_deg"])
    df.to_csv(OUT_CSV, index=False)
    print("Wrote", OUT_CSV)

    for r in rows:
        print(r)

    # recommend frames: convert Et -> frame index using CENTER_UTC_ISO / TOTALITY_FRAME mapping
    offset0 = center_et - (TOTALITY_FRAME / FPS)
    print("\nMapping: frame 0 ET = center_et - TOTALITY_FRAME/FPS =", offset0)
    def et_to_frame(et):
        if et is None:
            return None
        return int(round((et - offset0) * FPS))
    print("Recommended frame indices (approx):")
    print(" first_contact frame:", et_to_frame(et_first))
    print(" second_contact frame:", et_to_frame(et_second))
    print(" center frame:", et_to_frame(center_et))
    print(" third_contact frame:", et_to_frame(et_third))
    print(" fourth_contact frame:", et_to_frame(et_fourth))

if __name__ == "__main__":
    main()
