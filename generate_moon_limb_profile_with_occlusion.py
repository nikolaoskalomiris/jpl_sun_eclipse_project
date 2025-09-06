#!/usr/bin/env python3
"""
generate_moon_limb_profile_with_occlusion.py

Patched occlusion detection and GLD100-safe DEM sampling. This version accepts
command-line arguments so it can be called repeatedly by the orchestrator.

Requirements:
    pip install spiceypy rasterio numpy pandas

Usage examples:
    python generate_moon_limb_profile_with_occlusion.py --utc "2006-03-29 10:54:04.555" --out-csv out.csv
    python generate_moon_limb_profile_with_occlusion.py --utc "2006-03-29 10:54:04.555" --out-csv out.csv --preview-n-angles 256
"""

import os
import math
import time
import multiprocessing as mp
from multiprocessing import get_context
import argparse
import numpy as np
import pandas as pd
import spiceypy as sp
import rasterio
from rasterio.warp import transform as rio_transform
import rasterio.windows

# ----------------- USER CONFIGURATION (defaults) -----------------
# These defaults are preserved for users who run the script directly without args.
DEFAULT_KERNEL_DIR = r"C:\Users\SoR\Desktop\jpl_sun_eclipse_project\spice_kernels"
DEFAULT_DEM_PATH = r"C:\Users\SoR\Desktop\jpl_sun_eclipse_project\moon_dem\GLD100.tif"
DEFAULT_UTC = "2006-03-29 10:54:04.555"
DEFAULT_OBSERVER_LAT = 36.14265853184001
DEFAULT_OBSERVER_LON = 29.576375086997015
DEFAULT_OBSERVER_ALT = 2.0
DEFAULT_OUT_CSV = "moon_limb_profile.csv"

# supplied kernel files (these are attempted to be loaded if present)
KERNEL_FILES = [
    "naif0012.tls",
    "pck00010.tpc",
    "de440.bsp",
]

# camera model (for sanity checks)
sensor_width_mm = 21.44
sensor_pixels = 4096.0
mm_per_pixel = sensor_width_mm / sensor_pixels
f_mm = 35.0

# limb sampling default
DEFAULT_N_ANGLES = 2048

# ray marching (tweak these for resolution / correctness)
DEFAULT_RAY_STEP_KM = 0.2  # fine step (km)
DEFAULT_COARSE_FACTOR = 6  # coarse step = ray_step_km * coarse_factor
DEFAULT_EXTRA_CLEARANCE_KM = 5.0
EPS_KM = 1e-3  # 1 meter tolerance
MAX_RADIUS_PX = 1e6

# multiprocessing defaults
DEFAULT_USE_MULTIPROC = True
DEFAULT_NUM_WORKERS = max(1, min(mp.cpu_count() - 1, 11))

# GLD100-specific geospatial limits (from USGS GLD100 metadata)
DEM_MIN_LAT = -79.0
DEM_MAX_LAT = 79.0
DEM_MIN_LON = -180.0
DEM_MAX_LON = 180.0
DEM_LAT_EPS = 1e-8

# debugging flag
DEBUG_OCCLUSION = False
# ----------------- End defaults -----------------


def find_ephemeris_bsp(kernel_dir):
    for fn in os.listdir(kernel_dir):
        if fn.lower().startswith("de") and fn.lower().endswith(".bsp"):
            return os.path.join(kernel_dir, fn)
    return None


def load_spice_kernels(kernel_dir, extras=None):
    loaded = []
    if extras is None:
        extras = []
    # try preferred files first, then try any BSP/TLS in the folder
    for kg in KERNEL_FILES + extras:
        p = os.path.join(kernel_dir, kg)
        if os.path.exists(p):
            try:
                sp.furnsh(p)
                loaded.append(p)
            except Exception:
                pass
    bsp = find_ephemeris_bsp(kernel_dir)
    if bsp and bsp not in loaded:
        try:
            sp.furnsh(bsp)
            loaded.append(bsp)
        except Exception:
            pass
    # finally try to load any kernels in kernel_dir (best-effort)
    for fn in os.listdir(kernel_dir):
        if fn.lower().endswith((".tls", ".tpc", ".bsp", ".tf", ".tm")):
            p = os.path.join(kernel_dir, fn)
            if p not in loaded:
                try:
                    sp.furnsh(p)
                    loaded.append(p)
                except Exception:
                    pass
    return loaded


def dem_sample_point_ds(ds, lon_deg, lat_deg):
    """
    Bilinear sample DEM (handles wrap-around lon candidates).
    Returns elevation in meters or nan. GLD100 covers latitudes [-79, 79] (planetocentric).
    We clamp latitudes here.
    """
    lat_deg = float(lat_deg)
    lon_deg = float(lon_deg)
    lat_deg = max(DEM_MIN_LAT + DEM_LAT_EPS, min(DEM_MAX_LAT - DEM_LAT_EPS, lat_deg))
    try:
        nodata = ds.nodatavals[0]
    except Exception:
        nodata = None

    ds_crs = ds.crs
    lon_candidates = [lon_deg, lon_deg + 360.0, lon_deg - 360.0]
    for lon_try in lon_candidates:
        try:
            if ds_crs is None:
                xs, ys = lon_try, lat_deg
            else:
                xs_list, ys_list = rio_transform("EPSG:4326", ds_crs, [lon_try], [lat_deg])
                xs, ys = float(xs_list[0]), float(ys_list[0])
        except Exception:
            xs, ys = lon_try, lat_deg

        try:
            inv = ~ds.transform
            colf, rowf = inv * (xs, ys)
        except Exception:
            try:
                colf, rowf = ds.index(xs, ys, op=float)
            except Exception:
                continue

        if colf < -1 or colf > ds.width or rowf < -1 or rowf > ds.height:
            continue

        col0 = int(math.floor(colf)); row0 = int(math.floor(rowf))
        col1 = min(col0 + 1, ds.width - 1); row1 = min(row0 + 1, ds.height - 1)
        col0 = max(0, min(col0, ds.width - 1)); row0 = max(0, min(row0, ds.height - 1))

        try:
            window = rasterio.windows.Window(col_off=col0, row_off=row0, width=(col1-col0+1), height=(row1-row0+1))
            arr = ds.read(1, window=window, boundless=True, fill_value=(nodata if nodata is not None else np.nan))
        except Exception:
            try:
                v = ds.read(1, window=rasterio.windows.Window(col_off=col0, row_off=row0, width=1, height=1))
                v = float(v[0, 0])
                if nodata is not None and v == nodata:
                    return float('nan')
                return float(v)
            except Exception:
                continue

        vals = arr.astype(float)
        if nodata is not None:
            mask = (vals == nodata)
        else:
            mask = np.isnan(vals)

        if mask.all():
            continue
        if mask.any():
            good = vals[~mask]
            if good.size == 0:
                continue
            vals[mask] = float(np.mean(good))

        fx = colf - col0
        fy = rowf - row0
        fx = min(max(fx, 0.0), 1.0)
        fy = min(max(fy, 0.0), 1.0)

        h, w = vals.shape
        if h == 1 and w == 1:
            v = float(vals[0, 0])
            if nodata is not None and v == nodata:
                continue
            return float(v)

        if h == 1:
            vals = np.vstack([vals, vals])
        if w == 1:
            vals = np.hstack([vals, vals])

        v00 = float(vals[0, 0]); v10 = float(vals[0, 1])
        v01 = float(vals[1, 0]); v11 = float(vals[1, 1])
        v0 = v00 * (1 - fx) + v10 * fx
        v1 = v01 * (1 - fx) + v11 * fx
        v = v0 * (1 - fy) + v1 * fy
        return float(v)

    return float('nan')


# per-worker DEM init
_GLOBAL_DEM_PATH = None
_GLOBAL_DEM_DS = None


def _worker_init(dem_path):
    global _GLOBAL_DEM_PATH, _GLOBAL_DEM_DS
    _GLOBAL_DEM_PATH = dem_path
    try:
        _GLOBAL_DEM_DS = rasterio.open(_GLOBAL_DEM_PATH)
    except Exception as e:
        _GLOBAL_DEM_DS = None
        raise RuntimeError("Worker failed to open DEM at {}: {}".format(dem_path, e))


def find_intersection_for_psi(args):
    """
    Worker: compute limb sample for a single psi angle.
    Returns (idx, row_dict).
    """
    (idx, psi, moon_pos_wrt_earth, sun_pos_wrt_earth, obs_j2000, J2M, M2J, u_vec,
     moon_mean_radius_km, f_mm, mm_per_pixel,
     ray_step_km, coarse_factor, stop_radius_km, extra_clearance_km) = args

    ds = _GLOBAL_DEM_DS
    if ds is None:
        raise RuntimeError("DEM not initialized in worker")

    moon_pos = np.array(moon_pos_wrt_earth, dtype=float)
    sun_pos = np.array(sun_pos_wrt_earth, dtype=float)
    obs_pos = np.array(obs_j2000, dtype=float)
    J2M = np.array(J2M, dtype=float)
    M2J = np.array(M2J, dtype=float)
    u = np.array(u_vec, dtype=float)

    # build orthonormal basis e1,e2 perpendicular to u
    world_up = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(world_up, u)) > 0.9999:
        world_up = np.array([0.0, 1.0, 0.0])
    e1 = np.cross(world_up, u)
    e1n = np.linalg.norm(e1)
    if e1n < 1e-12:
        e1 = np.array([1.0, 0.0, 0.0])
    else:
        e1 = e1 / e1n
    e2 = np.cross(u, e1)
    e2 = e2 / np.linalg.norm(e2)

    dir_j2000 = math.cos(psi) * e1 + math.sin(psi) * e2

    # direction in moon-fixed frame -> get surface lat/lon from this direction
    dir_mf = J2M.dot(dir_j2000)
    x, y, z = float(dir_mf[0]), float(dir_mf[1]), float(dir_mf[2])
    r = math.sqrt(x * x + y * y + z * z)
    if r <= 0:
        lat_deg = 0.0; lon_deg = 0.0
    else:
        lat_rad = math.asin(z / r)
        lon_rad = math.atan2(y, x)
        lat_deg = math.degrees(lat_rad)
        lon_deg = math.degrees(lon_rad)
        if lon_deg > 180.0:
            lon_deg -= 360.0

    # Clamp lat/lon to GLD100 extents before sampling elevation
    lat_deg = max(DEM_MIN_LAT + DEM_LAT_EPS, min(DEM_MAX_LAT - DEM_LAT_EPS, lat_deg))
    if lon_deg > 180.0:
        lon_deg -= 360.0
    if lon_deg < -180.0:
        lon_deg += 360.0

    elev_m = dem_sample_point_ds(ds, lon_deg, lat_deg)
    if math.isnan(elev_m):
        elev_m = 0.0

    eff_r_km = float(moon_mean_radius_km + elev_m / 1000.0)

    # surface point in moon frame, convert to J2000 absolute
    surf_mf = np.array([x, y, z], dtype=float)
    normsurf = np.linalg.norm(surf_mf)
    if normsurf <= 0:
        surf_mf_unit = np.array([1.0, 0.0, 0.0])
    else:
        surf_mf_unit = surf_mf / normsurf
    surf_mf_pos = surf_mf_unit * eff_r_km
    surf_j2000 = M2J.dot(surf_mf_pos)
    surface_abs_j2000 = moon_pos + surf_j2000

    # angle between observer->surface and observer->moon center (for image radius)
    v_surf = surface_abs_j2000 - obs_pos
    v_center = moon_pos - obs_pos
    norm_vs = np.linalg.norm(v_surf)
    norm_vc = np.linalg.norm(v_center)
    if norm_vs <= 0 or norm_vc <= 0:
        ang_rad = 0.0
    else:
        dotv = float(np.dot(v_surf, v_center) / (norm_vs * norm_vc))
        dotv = max(-1.0, min(1.0, dotv))
        ang_rad = math.acos(dotv)

    # project to px (thin-lens / small-angle approx)
    try:
        radius_mm = f_mm * math.tan(ang_rad)
    except Exception:
        radius_mm = f_mm * ang_rad
    radius_px = radius_mm / mm_per_pixel
    if not np.isfinite(radius_px) or radius_px < 0:
        radius_px = 0.0
    radius_px = min(radius_px, MAX_RADIUS_PX)
    x_px = radius_px * math.cos(psi)
    y_px = radius_px * math.sin(psi)

    # --- Occlusion test: cast ray from surface towards Sun and look for intersection with lunar surface ---
    vec_sun = sun_pos - surface_abs_j2000
    dist_to_sun_km = np.linalg.norm(vec_sun)
    sun_visible = True
    if dist_to_sun_km <= 0:
        sun_visible = False
    else:
        sun_dir = vec_sun / dist_to_sun_km
        # safe sampling ranges
        s_start = max(ray_step_km * 0.05, 1e-6)  # start very close to surface
        max_s = min(dist_to_sun_km, stop_radius_km * 2.5)
        coarse_step = ray_step_km * coarse_factor
        if coarse_step < ray_step_km:
            coarse_step = ray_step_km
        hit_coarse = False
        last_safe_s = 0.0
        s = s_start
        # Coarse scanning (but starting very near surface)
        while s <= max_s:
            sample_pos = surface_abs_j2000 + s * sun_dir
            vec_from_moon_center = sample_pos - moon_pos
            r_km = np.linalg.norm(vec_from_moon_center)
            # If we're already far outside the moon, no occlusion further
            if r_km > (stop_radius_km + extra_clearance_km):
                break
            # convert to moon-fixed to sample DEM
            sample_mf = J2M.dot(vec_from_moon_center)
            sx, sy, sz = float(sample_mf[0]), float(sample_mf[1]), float(sample_mf[2])
            rr = math.sqrt(sx * sx + sy * sy + sz * sz)
            if rr <= 0:
                last_safe_s = s
                s += coarse_step
                continue
            sample_lat_rad = math.asin(max(-1.0, min(1.0, sz / rr)))
            sample_lon_rad = math.atan2(sy, sx)
            sample_lat_deg = math.degrees(sample_lat_rad)
            sample_lon_deg = math.degrees(sample_lon_rad)
            if sample_lon_deg > 180.0:
                sample_lon_deg -= 360.0
            # clamp sample latitude before DEM access
            sample_lat_deg = max(DEM_MIN_LAT + DEM_LAT_EPS, min(DEM_MAX_LAT - DEM_LAT_EPS, sample_lat_deg))
            sample_elev_m = dem_sample_point_ds(ds, sample_lon_deg, sample_lat_deg)
            if math.isnan(sample_elev_m):
                last_safe_s = s
                s += coarse_step
                continue
            sample_local_radius_km = float(moon_mean_radius_km + (sample_elev_m / 1000.0))
            # If the sample point is inside the local surface radius -> there is an occluding terrain
            if rr < (sample_local_radius_km - EPS_KM):
                hit_coarse = True
                hit_s = s
                break
            last_safe_s = s
            s += coarse_step

        if not hit_coarse:
            sun_visible = True
        else:
            # refine using smaller steps between last_safe_s and hit_s
            refine_start = max(ray_step_km * 0.01, last_safe_s)
            refine_end = max(hit_s, refine_start + ray_step_km)
            refine_step = max(ray_step_km * 0.125, 0.01)
            s_ref = refine_start
            occluded = False
            while s_ref <= refine_end:
                sample_pos = surface_abs_j2000 + s_ref * sun_dir
                vec_from_moon_center = sample_pos - moon_pos
                r_km = np.linalg.norm(vec_from_moon_center)
                if r_km > (stop_radius_km + extra_clearance_km):
                    break
                sample_mf = J2M.dot(vec_from_moon_center)
                sx, sy, sz = float(sample_mf[0]), float(sample_mf[1]), float(sample_mf[2])
                rr = math.sqrt(sx * sx + sy * sy + sz * sz)
                if rr <= 0:
                    s_ref += refine_step
                    continue
                sample_lat_rad = math.asin(max(-1.0, min(1.0, sz / rr)))
                sample_lon_rad = math.atan2(sy, sx)
                sample_lat_deg = math.degrees(sample_lat_rad)
                sample_lon_deg = math.degrees(sample_lon_rad)
                if sample_lon_deg > 180.0:
                    sample_lon_deg -= 360.0
                sample_lat_deg = max(DEM_MIN_LAT + DEM_LAT_EPS, min(DEM_MAX_LAT - DEM_LAT_EPS, sample_lat_deg))
                sample_elev_m = dem_sample_point_ds(ds, sample_lon_deg, sample_lat_deg)
                if math.isnan(sample_elev_m):
                    s_ref += refine_step
                    continue
                sample_local_radius_km = float(moon_mean_radius_km + (sample_elev_m / 1000.0))
                if rr < (sample_local_radius_km - EPS_KM):
                    occluded = True
                    break
                s_ref += refine_step
            sun_visible = not occluded

    # build output row
    row = {
        "psi_deg": float(math.degrees(psi)),
        "lat_deg": float(lat_deg),
        "lon_deg": float(lon_deg),
        "elev_m": float(elev_m),
        "eff_radius_km": float(eff_r_km),
        "ang_rad": float(ang_rad),
        "radius_px": float(radius_px),
        "x_px": float(x_px),
        "y_px": float(y_px),
        "sun_visible": bool(sun_visible),
    }

    if DEBUG_OCCLUSION and (idx % max(1, int(len(np.linspace(0, 2.0 * math.pi, 36))))) == 0:
        print(f"[DBG] psi_idx={idx} psi_deg={row['psi_deg']:.3f} lat={lat_deg:.3f} lon={lon_deg:.3f} elev_m={elev_m:.2f} sun_visible={sun_visible}")

    return (idx, row)


def parse_cli():
    p = argparse.ArgumentParser(description="generate_moon_limb_profile_with_occlusion.py")
    p.add_argument("--utc", dest="utc", default=DEFAULT_UTC, help="UTC time string (ISO-like).")
    p.add_argument("--out-csv", dest="out_csv", default=DEFAULT_OUT_CSV, help="Output CSV path.")
    p.add_argument("--n-angles", dest="n_angles", type=int, default=DEFAULT_N_ANGLES, help="Number of limb samples.")
    p.add_argument("--preview-n-angles", dest="preview_n_angles", type=int, default=None,
                   help="Shortcut to set a lower n_angles for fast preview runs (overrides --n-angles if provided).")
    p.add_argument("--dem-path", dest="dem_path", default=DEFAULT_DEM_PATH, help="DEM (GLD100) path.")
    p.add_argument("--kernel-dir", dest="kernel_dir", default=DEFAULT_KERNEL_DIR, help="SPICE kernels directory.")
    p.add_argument("--ray-step-km", dest="ray_step_km", type=float, default=DEFAULT_RAY_STEP_KM, help="Raymarching step in km.")
    p.add_argument("--coarse-factor", dest="coarse_factor", type=float, default=DEFAULT_COARSE_FACTOR, help="Coarse factor for raymarch.")
    p.add_argument("--extra-clearance-km", dest="extra_clearance_km", type=float, default=DEFAULT_EXTRA_CLEARANCE_KM, help="Extra clearance radius for raymarch.")
    p.add_argument("--n-workers", dest="n_workers", type=int, default=DEFAULT_NUM_WORKERS, help="Number of worker processes (when using multiprocessing).")
    p.add_argument("--no-multiproc", dest="no_multiproc", action="store_true", help="Disable multiprocessing (single-threaded).")
    p.add_argument("--observer-lat", dest="observer_lat", type=float, default=DEFAULT_OBSERVER_LAT, help="Observer latitude (deg).")
    p.add_argument("--observer-lon", dest="observer_lon", type=float, default=DEFAULT_OBSERVER_LON, help="Observer longitude (deg).")
    p.add_argument("--observer-alt", dest="observer_alt", type=float, default=DEFAULT_OBSERVER_ALT, help="Observer altitude (m).")
    args = p.parse_args()
    return args


def main():
    global DEBUG_OCCLUSION

    args = parse_cli()

    # Resolve effective parameters (CLI overrides defaults)
    t_utc_iso = args.utc
    OUT_CSV = args.out_csv
    n_angles = args.n_angles
    if args.preview_n_angles is not None:
        n_angles = args.preview_n_angles  # preview overrides full sampling
    DEM_PATH = args.dem_path
    KERNEL_DIR = args.kernel_dir
    ray_step_km = args.ray_step_km
    coarse_factor = args.coarse_factor
    extra_clearance_km = args.extra_clearance_km
    use_multiprocessing = (not args.no_multiproc)
    num_workers = max(1, args.n_workers) if use_multiprocessing else 1
    observer_lat_deg = args.observer_lat
    observer_lon_deg = args.observer_lon
    observer_alt_m = args.observer_alt

    print("=== generate_moon_limb_profile_with_occlusion.py (patched occlusion) ===")
    print("KERNEL_DIR:", KERNEL_DIR)
    print("DEM_PATH:", DEM_PATH)
    print("Time (UTC):", t_utc_iso)
    print("Observer lat,lon,alt:", observer_lat_deg, observer_lon_deg, observer_alt_m)

    if not os.path.isdir(KERNEL_DIR):
        raise RuntimeError("KERNEL_DIR not found: " + KERNEL_DIR)

    loaded = load_spice_kernels(KERNEL_DIR)
    print("\nSPICE kernels loaded ({}):".format(len(loaded)))
    for p in loaded:
        print(" ", p)
    bsp_used = None
    for p in loaded:
        fn = os.path.basename(p).lower()
        if fn.startswith("de") and fn.endswith(".bsp"):
            bsp_used = fn; break
    print("Ephemeris used:", bsp_used if bsp_used else "(none detected)")

    # Convert UTC to ET (SPICE)
    try:
        et = sp.str2et(t_utc_iso)
    except Exception as e:
        # If str2et fails, try wrapping in more ISO-like str
        raise RuntimeError(f"spice.str2et failed for UTC string '{t_utc_iso}': {e}")
    print("ET:", et)

    if not os.path.exists(DEM_PATH):
        raise RuntimeError("DEM not found: " + DEM_PATH)

    ds_local = rasterio.open(DEM_PATH)
    print("\nOpened DEM: size {} x {}".format(ds_local.width, ds_local.height))
    try:
        small = ds_local.read(1, out_shape=(min(1024, ds_local.height), min(1024, ds_local.width)))
        approx_max_elev_m = float(np.nanmax(small))
    except Exception:
        approx_max_elev_m = 5000.0
    print("Approx DEM max elevation (m):", approx_max_elev_m)
    print("DEM CRS:", ds_local.crs)
    print("DEM bounds:", ds_local.bounds)
    print("DEM transform:", ds_local.transform)
    print("DEM nodata:", ds_local.nodatavals)

    # observer in body-fixed converted to J2000
    try:
        _cnt, rvals = sp.bodvrd("EARTH", "RADII", 3)
        re_km = float(rvals[0]); rp_km = float(min(rvals))
    except Exception:
        re_km = 6378.137; rp_km = 6356.752

    lon_rad = math.radians(observer_lon_deg)
    lat_rad = math.radians(observer_lat_deg)
    obs_body_km = sp.georec(lon_rad, lat_rad, observer_alt_m / 1000.0, re_km, (re_km - rp_km) / re_km)
    obs_body_km = np.array(obs_body_km, dtype=float)

    frame_from = "ITRF93"
    try:
        xform = sp.pxform(frame_from, "J2000", et)
    except Exception:
        frame_from = "IAU_EARTH"
        xform = sp.pxform(frame_from, "J2000", et)
    obs_j2000 = sp.mxv(xform, obs_body_km)

    moon_pos_wrt_earth, _ = sp.spkpos("MOON", et, "J2000", "NONE", "EARTH")
    moon_pos_wrt_earth = np.array(moon_pos_wrt_earth, dtype=float)
    sun_pos_wrt_earth, _ = sp.spkpos("SUN", et, "J2000", "NONE", "EARTH")
    sun_pos_wrt_earth = np.array(sun_pos_wrt_earth, dtype=float)

    vec_moon_to_obs = obs_j2000 - moon_pos_wrt_earth
    u_moon_to_obs = vec_moon_to_obs / np.linalg.norm(vec_moon_to_obs)
    dist_moon_to_obs_km = np.linalg.norm(vec_moon_to_obs)

    print("\nMoon center (J2000) wrt Earth (km):", moon_pos_wrt_earth)
    print("Observer (J2000) wrt Earth (km):", obs_j2000)
    print("Distance Moon->Observer (km):", dist_moon_to_obs_km)

    _, moon_radii = sp.bodvrd("MOON", "RADII", 3)
    moon_radii = np.array(moon_radii, dtype=float)
    moon_mean_radius_km = float(np.mean(moon_radii))
    print("Moon mean radius (km):", moon_mean_radius_km)

    j2000_to_moon = sp.pxform("J2000", "IAU_MOON", et)
    moon_to_j2000 = sp.pxform("IAU_MOON", "J2000", et)
    J2M = np.array(j2000_to_moon, dtype=float)
    M2J = np.array(moon_to_j2000, dtype=float)

    max_elev_km = approx_max_elev_m / 1000.0
    stop_radius_km = moon_mean_radius_km + max_elev_km + extra_clearance_km

    print("\nSampling limb with n_angles =", n_angles)
    print("Ray-marching params: ray_step_km =", ray_step_km, "coarse_factor =", coarse_factor, "stop_radius_km =", stop_radius_km)

    psis = np.linspace(0.0, 2.0 * math.pi, n_angles, endpoint=False)
    # prepare worker args
    worker_args = []
    for idx, psi in enumerate(psis):
        worker_args.append((
            idx, psi,
            moon_pos_wrt_earth.tolist(),
            sun_pos_wrt_earth.tolist(),
            obs_j2000.tolist(),
            J2M.tolist(),
            M2J.tolist(),
            u_moon_to_obs.tolist(),
            moon_mean_radius_km,
            f_mm,
            mm_per_pixel,
            ray_step_km,
            coarse_factor,
            stop_radius_km,
            extra_clearance_km
        ))

    results = [None] * n_angles
    start_time = time.time()
    occluded_count = 0

    if use_multiprocessing and num_workers > 1:
        ctx = get_context("spawn")
        print("\nUsing multiprocessing: True num_workers:", num_workers)
        print("Starting Pool with {} workers...".format(num_workers))
        pool = ctx.Pool(processes=num_workers, initializer=_worker_init, initargs=(DEM_PATH,))
        try:
            it = pool.imap_unordered(find_intersection_for_psi, worker_args, chunksize=32)
            processed = 0
            last_print = time.time()
            for res in it:
                idx, row = res
                results[idx] = row
                if not row["sun_visible"]:
                    occluded_count += 1
                processed += 1
                if processed % max(1, n_angles // 100) == 0 or (time.time() - last_print) > 1.0:
                    elapsed = time.time() - start_time
                    rate = processed / elapsed if elapsed > 0 else 0.0
                    eta = (n_angles - processed) / rate if rate > 0 else float('inf')
                    print(f"Processed {processed}/{n_angles} angles... rate={rate:.2f}/s ETA={eta:.1f}s")
                    last_print = time.time()
        finally:
            pool.close()
            pool.join()
    else:
        print("\nUsing multiprocessing: False (running single-threaded)")
        _worker_init(DEM_PATH)
        processed = 0
        last_print = time.time()
        for args_w in worker_args:
            idx, row = find_intersection_for_psi(args_w)
            results[idx] = row
            if not row["sun_visible"]:
                occluded_count += 1
            processed += 1
            if processed % max(1, n_angles // 100) == 0 or (time.time() - last_print) > 1.0:
                elapsed = time.time() - start_time
                rate = processed / elapsed if elapsed > 0 else 0.0
                eta = (n_angles - processed) / rate if rate > 0 else float('inf')
                print(f"Processed {processed}/{n_angles} angles... rate={rate:.2f}/s ETA={eta:.1f}s")
                last_print = time.time()

    elapsed = time.time() - start_time
    print(f"\nProcessed {n_angles} angles in {elapsed:.1f}s ({elapsed/n_angles:.4f} s per angle)")
    print("Detected occluded (sun not visible) limb samples:", occluded_count, " / ", n_angles)

    rows_ordered = [results[i] for i in range(n_angles)]
    df = pd.DataFrame(rows_ordered)

    # Write CSV to the requested output path
    df.to_csv(OUT_CSV, index=False)
    print("\nWrote", OUT_CSV)
    try:
        min_r = float(np.nanmin(df['radius_px']))
        max_r = float(np.nanmax(df['radius_px']))
    except Exception:
        min_r = float('nan'); max_r = float('nan')
    print("Min/Max radius_px:", min_r, "/", max_r)
    visible_frac = float(df['sun_visible'].sum()) / len(df)
    print("Sun-visible fraction:", visible_frac)
    ds_local.close()
    print("Done.")


if __name__ == "__main__":
    main()
