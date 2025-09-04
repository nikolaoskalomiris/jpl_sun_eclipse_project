#!/usr/bin/env python3
"""
generate_moon_limb_profile_with_occlusion.py

Fixed / changed:
 - Correct angular-offset computation (observer -> surface vs observer -> moon center)
 - Avoids infinite/tiny numerical issues and clamps insane pixel radii
 - Multiprocessing (spawn) with per-worker DEM init
 - coarse->refine ray marching, progress/ETA printed
 - DEM sampling: clamps latitudes to GLD100 supported range (-79..79°) with tiny eps
 - Minor robustness guards in sampling and geometry

Requirements:
  pip install spiceypy rasterio numpy pandas
"""

import os
import math
import time
import multiprocessing as mp
from multiprocessing import get_context
import numpy as np
import pandas as pd
import spiceypy as sp
import rasterio
from rasterio.warp import transform as rio_transform
import rasterio.windows

# ----------------- USER CONFIGURATION -----------------
KERNEL_DIR = r"C:\Users\SoR\Desktop\jpl_sun_eclipse_project\spice_kernels"
DEM_PATH = r"C:\Users\SoR\Desktop\jpl_sun_eclipse_project\moon_dem\GLD100.tif"
t_utc_iso = "2006-03-29 10:53:22.600"
observer_lat_deg = 36.14265853184001
observer_lon_deg = 29.576375086997015
observer_alt_m = 2.0
OUT_CSV = "moon_limb_profile.csv"

KERNEL_FILES = [
    "naif0012.tls",
    "pck00010.tpc",
    "de440.bsp",
]

# camera model
sensor_width_mm = 21.44
sensor_pixels = 4096.0
mm_per_pixel = sensor_width_mm / sensor_pixels
f_mm = 35.0

# limb sampling
n_angles = 2048

# ray marching
ray_step_km = 0.2
coarse_factor = 8
extra_clearance_km = 5.0

use_multiprocessing = True
# pick workers: leave one core free
num_workers = max(1, min(mp.cpu_count() - 1, 11))

# clamps & eps
EPS_KM = 0.001  # 1 m tolerance
MAX_RADIUS_PX = 1e6  # clamp insane pixel radii

# --- GLD100-specific geospatial limits (from USGS GLD100 metadata) ---
# GLD100 covers latitudes from -79 to +79 degrees (planetocentric), longitudes -180..180
DEM_MIN_LAT = -79.0
DEM_MAX_LAT = 79.0
DEM_MIN_LON = -180.0
DEM_MAX_LON = 180.0
DEM_LAT_EPS = 1e-8  # small epsilon to avoid exact-edge numerical issues
# ------------------------------------------------------

def find_ephemeris_bsp(kernel_dir):
    for fn in os.listdir(kernel_dir):
        if fn.lower().startswith("de") and fn.lower().endswith(".bsp"):
            return os.path.join(kernel_dir, fn)
    return None


def load_spice_kernels(kernel_dir, extras=None):
    loaded = []
    if extras is None:
        extras = []
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
    Returns elevation in meters or nan.

    Important: GLD100 only covers latitudes in [-79, 79] (planetocentric).
    We'll clamp latitude to that range with a tiny epsilon to avoid
    attempting reads outside the raster vertical coverage.
    """
    # clamp lat to GLD100 supported range to avoid sampling outside DEM extents
    lat_deg = float(lat_deg)
    lon_deg = float(lon_deg)
    lat_deg = max(DEM_MIN_LAT + DEM_LAT_EPS, min(DEM_MAX_LAT - DEM_LAT_EPS, lat_deg))

    nodata = None
    try:
        nodata = ds.nodatavals[0]
    except Exception:
        nodata = None

    ds_crs = ds.crs
    # try candidate longitudes in case ds uses a different domain/wrap
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

        # if requested point is way outside raster, skip
        if colf < -1 or colf > ds.width or rowf < -1 or rowf > ds.height:
            continue

        col0 = int(math.floor(colf)); row0 = int(math.floor(rowf))
        col1 = min(col0 + 1, ds.width - 1)
        row1 = min(row0 + 1, ds.height - 1)
        col0 = max(0, min(col0, ds.width - 1))
        row0 = max(0, min(row0, ds.height - 1))

        try:
            window = rasterio.windows.Window(col_off=col0, row_off=row0,
                                             width=(col1-col0+1), height=(row1-row0+1))
            arr = ds.read(1, window=window, boundless=True,
                          fill_value=(nodata if nodata is not None else np.nan))
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
    args is a tuple of many precomputed constants (kept picklable). Returns (idx, row_dict).
    """
    (idx, psi, moon_pos_wrt_earth, sun_pos_wrt_earth, obs_j2000,
     J2M, M2J, u_vec, moon_mean_radius_km, f_mm, mm_per_pixel,
     ray_step_km, coarse_factor, stop_radius_km, extra_clearance_km) = args

    ds = _GLOBAL_DEM_DS
    if ds is None:
        raise RuntimeError("DEM not initialized in worker")

    # convert lists back to numpy arrays
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

    # direction in moon-fixed frame
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
    # dem_sample_point_ds also clamps, but we do a proactive clamp here for clarity
    lat_deg = max(DEM_MIN_LAT + DEM_LAT_EPS, min(DEM_MAX_LAT - DEM_LAT_EPS, lat_deg))
    # wrap longitude into -180..180
    if lon_deg > 180.0:
        lon_deg -= 360.0
    if lon_deg < -180.0:
        lon_deg += 360.0

    elev_m = dem_sample_point_ds(ds, lon_deg, lat_deg)
    if math.isnan(elev_m):
        # fallback to zero if no DEM data (should be rare after clamp)
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

    # --- Correct angular offset: angle between vectors from OBSERVER to surface and observer to moon center
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

    # compute radius in image mm->px (guard against extremely large values)
    try:
        radius_mm = f_mm * math.tan(ang_rad)
    except Exception:
        radius_mm = f_mm * ang_rad  # fallback
    radius_px = radius_mm / mm_per_pixel
    if not np.isfinite(radius_px) or radius_px < 0:
        radius_px = 0.0
    radius_px = min(radius_px, MAX_RADIUS_PX)
    x_px = radius_px * math.cos(psi)
    y_px = radius_px * math.sin(psi)

    # --- Occlusion test: coarse->refine ray-marching from surface toward Sun ---
    vec_sun = sun_pos - surface_abs_j2000
    dist_to_sun_km = np.linalg.norm(vec_sun)
    sun_visible = True
    if dist_to_sun_km <= 0:
        sun_visible = False
    else:
        sun_dir = vec_sun / dist_to_sun_km
        coarse_step = ray_step_km * coarse_factor
        s = coarse_step
        hit_coarse = False
        while True:
            sample_pos = surface_abs_j2000 + s * sun_dir
            vec_from_moon_center = sample_pos - moon_pos
            r_km = np.linalg.norm(vec_from_moon_center)
            if r_km > stop_radius_km:
                break
            sample_mf = J2M.dot(vec_from_moon_center)
            sx, sy, sz = float(sample_mf[0]), float(sample_mf[1]), float(sample_mf[2])
            rr = math.sqrt(sx*sx + sy*sy + sz*sz)
            if rr <= 0:
                s += coarse_step
                if s > stop_radius_km * 2.0:
                    break
                continue
            sample_lat_rad = math.asin(sz / rr)
            sample_lon_rad = math.atan2(sy, sx)
            sample_lat_deg = math.degrees(sample_lat_rad)
            sample_lon_deg = math.degrees(sample_lon_rad)
            if sample_lon_deg > 180.0:
                sample_lon_deg -= 360.0

            # clamp sample_lat_deg to GLD100 extent before DEM access
            sample_lat_deg = max(DEM_MIN_LAT + DEM_LAT_EPS, min(DEM_MAX_LAT - DEM_LAT_EPS, sample_lat_deg))

            sample_elev_m = dem_sample_point_ds(ds, sample_lon_deg, sample_lat_deg)
            if math.isnan(sample_elev_m):
                s += coarse_step
                continue
            sample_local_radius_km = float(moon_mean_radius_km + (sample_elev_m / 1000.0))
            if rr < (sample_local_radius_km - EPS_KM):
                hit_coarse = True
                break
            s += coarse_step
            if s > stop_radius_km * 2.0:
                break

        if not hit_coarse:
            sun_visible = True
        else:
            s_ref = max(ray_step_km, s - coarse_step)
            sun_visible = True
            while True:
                sample_pos = surface_abs_j2000 + s_ref * sun_dir
                vec_from_moon_center = sample_pos - moon_pos
                r_km = np.linalg.norm(vec_from_moon_center)
                if r_km > stop_radius_km:
                    break
                sample_mf = J2M.dot(vec_from_moon_center)
                sx, sy, sz = float(sample_mf[0]), float(sample_mf[1]), float(sample_mf[2])
                rr = math.sqrt(sx*sx + sy*sy + sz*sz)
                if rr <= 0:
                    s_ref += ray_step_km
                    if s_ref > stop_radius_km * 2.0:
                        break
                    continue
                sample_lat_rad = math.asin(sz / rr)
                sample_lon_rad = math.atan2(sy, sx)
                sample_lat_deg = math.degrees(sample_lat_rad)
                sample_lon_deg = math.degrees(sample_lon_rad)
                if sample_lon_deg > 180.0:
                    sample_lon_deg -= 360.0

                # clamp sample_lat_deg again
                sample_lat_deg = max(DEM_MIN_LAT + DEM_LAT_EPS, min(DEM_MAX_LAT - DEM_LAT_EPS, sample_lat_deg))

                sample_elev_m = dem_sample_point_ds(ds, sample_lon_deg, sample_lat_deg)
                if math.isnan(sample_elev_m):
                    s_ref += ray_step_km
                    continue
                sample_local_radius_km = float(moon_mean_radius_km + (sample_elev_m / 1000.0))
                if rr < (sample_local_radius_km - EPS_KM):
                    sun_visible = False
                    break
                s_ref += ray_step_km
                if s_ref > stop_radius_km * 2.0:
                    break

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
    return (idx, row)


def main():
    print("=== generate_moon_limb_profile_with_occlusion.py (patched) ===")
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
    et = sp.str2et(t_utc_iso)
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

    # observer J2000
    try:
        _cnt, rvals = sp.bodvrd("EARTH", "RADII", 3)
        re_km = float(rvals[0])
        rp_km = float(min(rvals))
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
            moon_pos_wrt_earth.tolist(), sun_pos_wrt_earth.tolist(),
            obs_j2000.tolist(),
            J2M.tolist(), M2J.tolist(), u_moon_to_obs.tolist(),
            moon_mean_radius_km, f_mm, mm_per_pixel, ray_step_km, coarse_factor,
            stop_radius_km, extra_clearance_km
        ))

    results = [None] * n_angles
    start_time = time.time()

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
        for args in worker_args:
            idx, row = find_intersection_for_psi(args)
            results[idx] = row
            processed += 1
            if processed % max(1, n_angles // 100) == 0 or (time.time() - last_print) > 1.0:
                elapsed = time.time() - start_time
                rate = processed / elapsed if elapsed > 0 else 0.0
                eta = (n_angles - processed) / rate if rate > 0 else float('inf')
                print(f"Processed {processed}/{n_angles} angles... rate={rate:.2f}/s ETA={eta:.1f}s")
                last_print = time.time()

    elapsed = time.time() - start_time
    print(f"\nProcessed {n_angles} angles in {elapsed:.1f}s ({elapsed/n_angles:.4f} s per angle)")

    rows_ordered = [results[i] for i in range(n_angles)]
    df = pd.DataFrame(rows_ordered)
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
