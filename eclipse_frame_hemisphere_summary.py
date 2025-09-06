#!/usr/bin/env python3
"""
eclipse_frame_hemisphere_summary.py (patch: use sun position relative to MOON)

Samples the visible hemisphere (sub-observer cap) instead of only the rim.
Writes frame_summary.csv with three visibility fractions:
 - fraction_center_visible : Sun center above local horizon
 - fraction_any_visible    : any part of solar disk above horizon
 - fraction_fully_visible  : entire solar disk above horizon

Key change: compute Sun vector *relative to the Moon* (sp.spkpos(..., 'MOON')) to avoid
small cancellation errors when subtracting large Earth-centered vectors.
"""

import os
import math
import numpy as np
import pandas as pd
import spiceypy as sp

# ---------------- USER CONFIGURATION ----------------
KERNEL_DIR = r"C:\Users\SoR\Desktop\jpl_sun_eclipse_project\spice_kernels"
TOTALITY_UTC_ISO = "2006-03-29 10:54:04.555"

OBSERVER_LAT_DEG = 36.14265853184001
OBSERVER_LON_DEG = 29.576375086997015
OBSERVER_ALT_M = 2.0

FRAMES = 2000
FPS = 25.0
TOTALITY_FRAME = 1000

# Hemisphere sampling resolution: n_theta * n_phi samples over hemisphere
N_THETA = 40   # radial resolution (0..pi/2)
N_PHI = 80     # azimuthal resolution

OUT_SUMMARY_CSV = "frame_summary.csv"
KERNEL_FILES = ["naif0012.tls","pck00010.tpc","de440.bsp"]
DEBUG = False
# ---------------- END USER CONFIGURATION ----------------


def load_kernels(kernel_dir):
    if not os.path.isdir(kernel_dir):
        raise RuntimeError("KERNEL_DIR does not exist: " + str(kernel_dir))
    count = 0
    for fn in os.listdir(kernel_dir):
        if fn.lower().endswith((".bsp", ".tls", ".tpc", ".tf", ".tm")):
            p = os.path.join(kernel_dir, fn)
            try:
                sp.furnsh(p)
                count += 1
            except Exception:
                pass
    return count


def observer_j2000(et):
    # convert geodetic to J2000 using Earth frames (like your heavy script)
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
    """
    Sun position provided as vector from Moon to Sun (km). Compute angular radius.
    """
    try:
        rvals = sp.bodvrd("SUN", "RADII", 3)[1]
        r_sun_km = float(rvals[0])
    except Exception:
        r_sun_km = 695700.0
    dist_km = float(np.linalg.norm(sun_pos_wrt_moon))
    if dist_km <= 0:
        return 0.0
    ratio = r_sun_km / dist_km
    if ratio >= 1.0:
        return math.pi/2.0
    return math.asin(max(-1.0, min(1.0, ratio)))


def hemisphere_sample_unit_vectors(u, n_theta, n_phi):
    """
    Build unit surface normals across the hemisphere centered on direction u (unit, J2000).
    Returns array shape (N,3) of unit normals in J2000.
    Area-preserving theta spacing via cos-grid.
    """
    # orthonormal basis e1, e2 perpendicular to u
    world_up = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(world_up, u)) > 0.9999:
        world_up = np.array([0.0, 1.0, 0.0])
    e1 = np.cross(world_up, u)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(u, e1)
    e2 /= np.linalg.norm(e2)

    cos_grid = np.linspace(1.0, 0.0, n_theta, endpoint=False)
    thetas = np.arccos(cos_grid)  # theta in [0, pi/2)
    phis = np.linspace(0.0, 2.0 * math.pi, n_phi, endpoint=False)

    vecs = []
    for th in thetas:
        st = math.sin(th); ct = math.cos(th)
        for ph in phis:
            v = ct * u + st * (math.cos(ph) * e1 + math.sin(ph) * e2)
            v = v / np.linalg.norm(v)
            vecs.append(v)
    return np.array(vecs, dtype=float)


def compute_frame_metrics(et, n_theta, n_phi):
    """
    Compute frame metrics using:
      - moon_pos_wrt_earth: for observer vector u = obs_j2000 - moon_pos_wrt_earth
      - sun_pos_wrt_moon: direct SPICE vector moon->sun (no subtraction)
    Returns fractions and diagnostics.
    """
    # moon position wrt Earth (for u)
    moon_pos_wrt_earth, _ = sp.spkpos("MOON", et, "J2000", "NONE", "EARTH")
    moon_pos_wrt_earth = np.array(moon_pos_wrt_earth, dtype=float)

    # sun position wrt MOON (direct vector moon->sun)
    sun_pos_wrt_moon, _ = sp.spkpos("SUN", et, "J2000", "NONE", "MOON")
    sun_pos_wrt_moon = np.array(sun_pos_wrt_moon, dtype=float)

    # observer in J2000
    obs_j2000 = observer_j2000(et)

    # u = direction from moon center to observer (J2000)
    u = obs_j2000 - moon_pos_wrt_earth
    un = np.linalg.norm(u)
    if un <= 0:
        raise RuntimeError("Bad observer/moon geometry; zero-length u")
    u = u / un

    # sample hemisphere normals oriented around u
    normals = hemisphere_sample_unit_vectors(u, n_theta, n_phi)
    Ns = normals.shape[0]

    # sun direction (moon->sun)
    sd_norm = np.linalg.norm(sun_pos_wrt_moon)
    if sd_norm <= 0:
        # degenerate: all visible
        return 1.0, 1.0, 1.0, {"dot_min": 1.0, "dot_mean": 1.0, "dot_max": 1.0}, 0.0, 0.0

    sun_dir_j2000 = sun_pos_wrt_moon / sd_norm

    # angular diagnostic: angle between sun_dir and u (useful sanity check)
    dot_sun_u = float(np.dot(sun_dir_j2000, u))
    ang_between_deg = math.degrees(math.acos(max(-1.0, min(1.0, dot_sun_u))))

    # solar angular radius as seen from Moon
    Rsun_rad = solar_angular_radius_rad_from_moon(sun_pos_wrt_moon)
    sinR = math.sin(Rsun_rad)

    # dot products
    dots = normals.dot(sun_dir_j2000)
    dot_min = float(np.min(dots)); dot_max = float(np.max(dots)); dot_mean = float(np.mean(dots))

    center_vis = np.sum(dots > 0.0)
    any_vis = np.sum(dots >= -sinR)
    full_vis = np.sum(dots >= sinR)

    return (float(center_vis)/Ns, float(any_vis)/Ns, float(full_vis)/Ns,
            {"dot_min":dot_min, "dot_mean":dot_mean, "dot_max":dot_max},
            math.degrees(Rsun_rad), ang_between_deg)


def main():
    print("=== eclipse_frame_hemisphere_summary.py (sun relative-to-moon patch) ===")
    print("Loading SPICE kernels from", KERNEL_DIR)
    n = load_kernels(KERNEL_DIR)
    print("Loaded (attempted) {} kernels.".format(n))

    et_total = sp.str2et(TOTALITY_UTC_ISO)
    start_et = et_total - (float(TOTALITY_FRAME) / float(FPS))
    print("Frame 0 ET:", start_et)
    total_samples = N_THETA * N_PHI
    print(f"Sampling hemisphere with {N_THETA} x {N_PHI} = {total_samples} points per frame.")

    rows = []
    for frame in range(FRAMES):
        et = start_et + (frame / float(FPS))
        frac_center, frac_any, frac_full, diag, solar_radius_deg, ang_between_deg = compute_frame_metrics(et, N_THETA, N_PHI)
        utc_iso = sp.et2utc(et, "C", 3)
        rows.append((frame, utc_iso, et, frac_center, frac_any, frac_full, total_samples,
                     diag["dot_min"], diag["dot_mean"], diag["dot_max"], solar_radius_deg, ang_between_deg))

        # print diagnostics for frames of interest or coarse progress
        if frame % max(1, FRAMES//20) == 0 or DEBUG or abs(frame - TOTALITY_FRAME) <= 5:
            print(f"frame {frame}/{FRAMES} center={frac_center:.4f} any={frac_any:.4f} full={frac_full:.4f} "
                  f"dot_min={diag['dot_min']:.6g} dot_mean={diag['dot_mean']:.6g} dot_max={diag['dot_max']:.6g} "
                  f"solar_radius_deg={solar_radius_deg:.6f} angle_sun_vs_u_deg={ang_between_deg:.6f}")

    df = pd.DataFrame(rows, columns=[
        "frame","utc_iso","et",
        "fraction_center_visible","fraction_any_visible","fraction_fully_visible",
        "n_samples","dot_min","dot_mean","dot_max","solar_radius_deg","angle_sun_vs_u_deg"
    ])
    df.to_csv(OUT_SUMMARY_CSV, index=False)
    print("Wrote", OUT_SUMMARY_CSV)
    print("Done.")


if __name__ == "__main__":
    main()
