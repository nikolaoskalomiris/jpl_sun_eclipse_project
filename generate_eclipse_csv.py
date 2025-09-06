#!/usr/bin/env python3
"""
generate_eclipse_csv.py (fixed + center_row_index)
- Writes eclipse_keyframes_full.csv with both time_s_center and time_s_from_start
- Writes center_metadata.json with authoritative center_utc/center_et/half_window/frames/fps/time_compression/ae_center_frame
- Adds center_row_index and center_row_time_s_center so downstream tools can align deterministically.
"""
import os, sys, math, json, argparse
import numpy as np
import pandas as pd
from astropy.time import Time
from astropy.coordinates import EarthLocation, AltAz, get_sun
import astropy.units as u

# moon function: prefer get_moon, fallback to get_body
try:
    from astropy.coordinates import get_moon
    _HAVE_GET_MOON = True
except Exception:
    from astropy.coordinates import get_body
    _HAVE_GET_MOON = False

# optional SPICE
try:
    import spiceypy as sp
    _HAVE_SPICE = True
except Exception:
    _HAVE_SPICE = False

# ---------------- USER PARAMETERS (defaults) ----------------
FRAMES = 2000
INITIAL_TOTAL_DURATION_SECONDS = 3.0 * 3600.0  # 3 hours
OBSERVER_LAT_DEG = 36.14265853184001
OBSERVER_LON_DEG = 29.576375086997015
OBSERVER_ALT_M = 2.0
START_UTC_ISO = "2006-03-29 10:54:04.557"  # initial guess
SEARCH_SPAN_SECONDS = INITIAL_TOTAL_DURATION_SECONDS
FPS_DEFAULT = 25.0
OUT_CSV_DEFAULT = "eclipse_keyframes_full.csv"
METADATA_JSON = "center_metadata.json"
AE_CENTER_FRAME_DEFAULT = 1000
SPICE_KERNEL_DIR_DEFAULT = "spice_kernels"
# Camera / imaging defaults (keeps previous semantics)
SENSOR_WIDTH_MM = 21.44
SENSOR_PIXELS = 4096.0
MM_PER_PIXEL = SENSOR_WIDTH_MM / SENSOR_PIXELS
F_MM = 35.0
MOON_DIAM_KM = 3474.8
SUN_DIAM_KM = 1391000.0
# -----------------------------------------------------------

# ---------------- helper functions ----------------
def ang_sep_astropy_time(t_astropy, location):
    altaz = AltAz(obstime=t_astropy, location=location)
    sun_topo = get_sun(t_astropy).transform_to(altaz)
    if _HAVE_GET_MOON:
        moon_topo = get_moon(t_astropy).transform_to(altaz)
    else:
        moon_topo = get_body('moon', t_astropy).transform_to(altaz)
    return float(sun_topo.separation(moon_topo).to(u.rad).value)

def furnsh_kernels(kernel_dir):
    loaded = []
    if not kernel_dir or not os.path.isdir(kernel_dir):
        return loaded
    # TLS first (leapseconds)
    for fn in os.listdir(kernel_dir):
        if fn.lower().endswith(".tls"):
            p = os.path.join(kernel_dir, fn)
            try:
                import spiceypy as sp
                sp.furnsh(p)
                loaded.append(p)
                print("Furnished TLS:", fn)
            except Exception as e:
                print("Warning: failed to furnsh tls:", p, "->", e)
    # BSPs
    for fn in os.listdir(kernel_dir):
        if fn.lower().endswith(".bsp"):
            p = os.path.join(kernel_dir, fn)
            try:
                import spiceypy as sp
                sp.furnsh(p)
                loaded.append(p)
                print("Furnished BSP:", fn)
            except Exception as e:
                print("Warning: failed to furnsh bsp:", p, "->", e)
    # other kernel types (optional)
    for fn in sorted(os.listdir(kernel_dir)):
        if fn.lower().endswith((".tpc", ".tm", ".tf")):
            p = os.path.join(kernel_dir, fn)
            try:
                import spiceypy as sp
                sp.furnsh(p)
                loaded.append(p)
            except Exception:
                pass
    return loaded

def ang_sep_spice_et(et, observer_lat_deg, observer_lon_deg, observer_alt_m):
    import numpy as np, math
    import spiceypy as sp
    sun_pos, _ = sp.spkpos("SUN", float(et), "J2000", "NONE", "EARTH")
    moon_pos, _ = sp.spkpos("MOON", float(et), "J2000", "NONE", "EARTH")
    sun_pos = np.array(sun_pos, dtype=float)
    moon_pos = np.array(moon_pos, dtype=float)
    try:
        radvals = sp.bodvrd("EARTH", "RADII", 3)[1]
        re_km = float(radvals[0]); rp_km = float(min(radvals))
    except Exception:
        re_km = 6378.137; rp_km = 6356.752
    lon_rad = math.radians(observer_lon_deg)
    lat_rad = math.radians(observer_lat_deg)
    alt_km = observer_alt_m / 1000.0
    flatten = (re_km - rp_km) / re_km if re_km != 0 else 0.0
    try:
        obs_body = sp.georec(lon_rad, lat_rad, alt_km, re_km, flatten)
    except Exception:
        obs_body = [0.0, 0.0, 0.0]
    frame_from = "ITRF93"
    try:
        xform = sp.pxform(frame_from, "J2000", float(et))
    except Exception:
        frame_from = "IAU_EARTH"
        xform = sp.pxform(frame_from, "J2000", float(et))
    obs_j2000 = sp.mxv(xform, obs_body)
    obs_j2000 = np.array(obs_j2000, dtype=float)
    sun_vec = sun_pos - obs_j2000
    moon_vec = moon_pos - obs_j2000
    us = sun_vec / np.linalg.norm(sun_vec)
    um = moon_vec / np.linalg.norm(moon_vec)
    dot = float(np.dot(us, um)); dot = max(-1.0, min(1.0, dot))
    return math.acos(dot)

def refine_min_sep_spice(center_et, half_window_s, obs_lat, obs_lon, obs_alt, coarse_step_s=1.0, tol_s=1e-3, max_iter=200):
    import numpy as np, math
    t0 = center_et - half_window_s
    t1 = center_et + half_window_s
    step = max(1.0, coarse_step_s)
    ts = np.arange(t0, t1 + 0.5*step, step)
    seps = np.array([ang_sep_spice_et(t, obs_lat, obs_lon, obs_alt) for t in ts])
    idx = int(np.nanargmin(seps))
    left_idx = max(0, idx - 8)
    right_idx = min(len(ts) - 1, idx + 8)
    a = float(ts[left_idx]); b = float(ts[right_idx])
    phi = (1.0 + 5.0**0.5) / 2.0; invphi = 1.0 / phi
    c = b - invphi * (b - a); d = a + invphi * (b - a)
    fc = ang_sep_spice_et(c, obs_lat, obs_lon, obs_alt); fd = ang_sep_spice_et(d, obs_lat, obs_lon, obs_alt)
    iter_count = 0
    while abs(b - a) > tol_s and iter_count < max_iter:
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - invphi * (b - a)
            fc = ang_sep_spice_et(c, obs_lat, obs_lon, obs_alt)
        else:
            a, c, fc = c, d, fd
            d = a + invphi * (b - a)
            fd = ang_sep_spice_et(d, obs_lat, obs_lon, obs_alt)
        iter_count += 1
    best_et = 0.5 * (a + b)
    best_sep = ang_sep_spice_et(best_et, obs_lat, obs_lon, obs_alt)
    return best_et, best_sep

def refine_min_sep_astropy(center_time, half_window_s, location, coarse_step_s=1.0, tol_s=1e-3, max_iter=200):
    import numpy as np, math
    t0 = center_time - (half_window_s * u.s)
    t1 = center_time + (half_window_s * u.s)
    N = max(101, int(math.ceil((2.0*half_window_s)/coarse_step_s)) + 1)
    ts = t0 + (np.linspace(0.0, 2.0*half_window_s, N) - half_window_s) * u.s
    seps = np.array([ang_sep_astropy_time(tt, location) for tt in ts])
    idx = int(np.nanargmin(seps))
    left_idx = max(0, idx - 8)
    right_idx = min(len(ts) - 1, idx + 8)
    a_rel = (ts[left_idx] - center_time).to_value(u.s)
    b_rel = (ts[right_idx] - center_time).to_value(u.s)
    phi = (1.0 + 5.0**0.5) / 2.0; invphi = 1.0/phi
    c_rel = b_rel - invphi * (b_rel - a_rel); d_rel = a_rel + invphi * (b_rel - a_rel)
    def sep_of_rel(s_rel):
        t_abs = center_time + (s_rel * u.s)
        return ang_sep_astropy_time(t_abs, location)
    fc = sep_of_rel(c_rel); fd = sep_of_rel(d_rel)
    iter_count = 0
    while abs(b_rel - a_rel) > tol_s and iter_count < max_iter:
        if fc < fd:
            b_rel = d_rel; d_rel = c_rel; fd = fc
            c_rel = b_rel - invphi * (b_rel - a_rel); fc = sep_of_rel(c_rel)
        else:
            a_rel = c_rel; c_rel = d_rel; fc = fd
            d_rel = a_rel + invphi * (b_rel - a_rel); fd = sep_of_rel(d_rel)
        iter_count += 1
    best_rel = 0.5 * (a_rel + b_rel)
    best_time = center_time + (best_rel * u.s)
    best_sep = ang_sep_astropy_time(best_time, location)
    return best_time, best_sep

# ---------------- main ----------------
def main():
    p = argparse.ArgumentParser(description="generate_eclipse_csv.py (fixed + center_row_index)")
    p.add_argument("--start-utc", dest="start_utc", type=str, default=START_UTC_ISO)
    p.add_argument("--frames", dest="frames", type=int, default=FRAMES)
    p.add_argument("--out-csv", dest="out_csv", type=str, default=OUT_CSV_DEFAULT, help="Output CSV path")
    p.add_argument("--kernel-dir", dest="kernel_dir", type=str, default=SPICE_KERNEL_DIR_DEFAULT, help="SPICE kernel dir (optional)")
    p.add_argument("--fps", dest="fps", type=float, default=FPS_DEFAULT)
    p.add_argument("--ae-center-frame", dest="ae_center_frame", type=int, default=AE_CENTER_FRAME_DEFAULT)
    args = p.parse_args()

    frames_local = int(args.frames)
    out_csv_local = str(args.out_csv)
    start_utc_local = str(args.start_utc)
    kernel_dir = args.kernel_dir
    fps = float(args.fps)
    ae_center_frame = int(args.ae_center_frame)

    center_time_guess = Time(start_utc_local, scale="utc")
    search_N = 4001
    search_times = center_time_guess + (np.linspace(-SEARCH_SPAN_SECONDS, SEARCH_SPAN_SECONDS, search_N) * u.s)

    loc = EarthLocation(lat=OBSERVER_LAT_DEG*u.deg, lon=OBSERVER_LON_DEG*u.deg, height=OBSERVER_ALT_M*u.m)

    seps = np.empty(search_N, dtype=float)
    for i, tt in enumerate(search_times):
        seps[i] = ang_sep_astropy_time(tt, loc)

    idx_min = int(np.argmin(seps))
    t_center_coarse = search_times[idx_min]
    min_sep_deg_coarse = (seps[idx_min] * u.rad).to(u.deg).value
    print("Found coarse minimum separation at:", t_center_coarse.iso, "UTC")
    print("Coarse minimum angular separation (deg):", min_sep_deg_coarse)

    # calculate approximate radii for edge threshold
    altaz_center = AltAz(obstime=t_center_coarse, location=loc)
    sun_center = get_sun(t_center_coarse).transform_to(altaz_center)
    if _HAVE_GET_MOON:
        moon_center = get_moon(t_center_coarse).transform_to(altaz_center)
    else:
        moon_center = get_body('moon', t_center_coarse).transform_to(altaz_center)
    dist_moon_m = float(moon_center.cartesian.norm().to(u.m).value)
    dist_sun_m = float(sun_center.cartesian.norm().to(u.m).value)
    theta_moon_center_rad = 2.0 * math.atan((MOON_DIAM_KM * 1000.0 / 2.0) / dist_moon_m)
    theta_sun_center_rad = 2.0 * math.atan((SUN_DIAM_KM * 1000.0 / 2.0) / dist_sun_m)
    moon_radius_center_deg = (theta_moon_center_rad / 2.0) * (180.0 / math.pi)
    sun_radius_center_deg = (theta_sun_center_rad / 2.0) * (180.0 / math.pi)
    touch_threshold_deg = (moon_radius_center_deg + sun_radius_center_deg) * 0.9

    half = INITIAL_TOTAL_DURATION_SECONDS / 2.0
    while True:
        t_edge_before = t_center_coarse - (half * u.s)
        t_edge_after = t_center_coarse + (half * u.s)
        sep_before = ang_sep_astropy_time(t_edge_before, loc)
        sep_after = ang_sep_astropy_time(t_edge_after, loc)
        sep_before_deg = math.degrees(sep_before) if isinstance(sep_before, float) else (sep_before * u.rad).to(u.deg).value
        sep_after_deg = math.degrees(sep_after) if isinstance(sep_after, float) else (sep_after * u.rad).to(u.deg).value
        min_edge_sep = min(sep_before_deg, sep_after_deg)
        print(f"Testing half-window = {half/3600.0:.3f} h -> edge seps (deg): before={sep_before_deg:.6f}, after={sep_after_deg:.6f}; threshold={touch_threshold_deg:.6f}")
        if min_edge_sep > touch_threshold_deg:
            break
        if half * 1.5 > 24.0 * 3600.0:
            print("Reached maximum expansion.")
            break
        half *= 1.5

    # SPICE refinement if available + kernels furnished
    refined_time_astropy = None
    refined_sep_deg = None
    center_et = None
    used_spice = False
    if _HAVE_SPICE and kernel_dir and os.path.isdir(kernel_dir):
        loaded = furnsh_kernels(kernel_dir)
        bsp_found = any(p.lower().endswith(".bsp") for p in loaded)
        tls_found = any(p.lower().endswith(".tls") for p in loaded)
        if not tls_found:
            print("Warning: no TLS found among furnished kernels (leapseconds).")
        if bsp_found:
            try:
                et_guess = sp.str2et(t_center_coarse.iso)
                et_min, sep_rad_min = refine_min_sep_spice(et_guess, min(half, 1800.0), OBSERVER_LAT_DEG, OBSERVER_LON_DEG, OBSERVER_ALT_M)
                center_et = float(et_min)
                refined_sep_deg = float(math.degrees(sep_rad_min))
                try:
                    utc_isot = sp.et2utc(center_et, 'ISOC', 3)
                    refined_time_astropy = Time(utc_isot, format='isot', scale='utc')
                except Exception:
                    refined_time_astropy = Time(t_center_coarse)
                used_spice = True
                print("Refined center ET (spice):", center_et)
                print("Refined minimum angular separation (deg):", refined_sep_deg)
            except Exception as e:
                print("SPICE refinement failed:", e)
                used_spice = False
        else:
            print("SPICE BSP not found; skipping SPICE refinement.")
    else:
        if not _HAVE_SPICE:
            print("spiceypy not available; skipping SPICE refinement.")
        else:
            print("kernel_dir missing; skipping SPICE refinement.")

    if not used_spice:
        try:
            print("Running astropy-based golden-section refinement...")
            t_refined, sep_rad_refined = refine_min_sep_astropy(t_center_coarse, min(half, 1800.0), loc, coarse_step_s=1.0, tol_s=1e-3)
            refined_time_astropy = t_refined
            refined_sep_deg = math.degrees(sep_rad_refined)
            print("Refined center (astropy):", refined_time_astropy.iso)
            print("Refined minimum angular separation (deg):", refined_sep_deg)
            if _HAVE_SPICE and kernel_dir and os.path.isdir(kernel_dir):
                try:
                    center_et = sp.str2et(refined_time_astropy.iso)
                    print("Converted refined UTC to ET:", center_et)
                except Exception as e:
                    print("Could not convert astropy refined UTC to ET via spice:", e)
        except Exception as e:
            print("Astropy refinement failed:", e)
            refined_time_astropy = t_center_coarse
            refined_sep_deg = min_sep_deg_coarse

    if refined_time_astropy is None:
        refined_time_astropy = t_center_coarse

    # final times for frames
    final_offsets = np.linspace(-half, half, frames_local)  # seconds from center
    final_times = refined_time_astropy + (final_offsets * u.s)

    rows = []
    sep_offsets = []
    for idx, t in enumerate(final_times):
        altaz_frame = AltAz(obstime=t, location=loc)
        sun_topo = get_sun(t).transform_to(altaz_frame)
        if _HAVE_GET_MOON:
            moon_topo = get_moon(t).transform_to(altaz_frame)
        else:
            moon_topo = get_body('moon', t).transform_to(altaz_frame)

        sun_cart = sun_topo.cartesian.xyz.to(u.m).value
        moon_cart = moon_topo.cartesian.xyz.to(u.m).value
        fwd = sun_cart
        fwd_norm = np.linalg.norm(fwd)
        if fwd_norm == 0:
            ex = np.array([1.0, 0.0, 0.0]); ey = np.array([0.0, 1.0, 0.0]); ez = np.array([0.0, 0.0, 1.0])
        else:
            ex = fwd / fwd_norm
            ref_up = np.array([0.0, 0.0, 1.0])
            right = np.cross(ref_up, ex)
            rn = np.linalg.norm(right)
            if rn < 1e-12:
                ref_up = np.array([0.0, 1.0, 0.0]); right = np.cross(ref_up, ex); rn = np.linalg.norm(right)
            if rn < 1e-12:
                right = np.array([1.0, 0.0, 0.0]); rn = 1.0
            ey = right / rn
            ez = np.cross(ex, ey)

        mdot_x = float(np.dot(moon_cart, ex)); mdot_y = float(np.dot(moon_cart, ey)); mdot_z = float(np.dot(moon_cart, ez))
        alpha = math.atan2(mdot_y, mdot_x); beta = math.atan2(mdot_z, mdot_x)
        ang_sep = math.atan2(math.sqrt(mdot_y*mdot_y + mdot_z*mdot_z), mdot_x)
        dist_moon_m = float(np.linalg.norm(moon_cart)); dist_sun_m = float(np.linalg.norm(sun_cart))
        theta_moon = 2.0 * math.atan((MOON_DIAM_KM * 1000.0 / 2.0) / dist_moon_m)
        theta_sun = 2.0 * math.atan((SUN_DIAM_KM * 1000.0 / 2.0) / dist_sun_m)
        moon_image_mm = 2.0 * F_MM * math.tan(theta_moon / 2.0)
        sun_image_mm = 2.0 * F_MM * math.tan(theta_sun / 2.0)
        moon_px = moon_image_mm / MM_PER_PIXEL
        sun_px = sun_image_mm / MM_PER_PIXEL
        sep_x_px = (F_MM * math.tan(alpha)) / MM_PER_PIXEL
        sep_y_px = (F_MM * math.tan(beta)) / MM_PER_PIXEL

        ts_center = float(final_offsets[idx])
        ts_from_start = ts_center + half

        row = {
            "frame": int(idx),
            "time_s_center": ts_center,
            "time_s_from_start": ts_from_start,
            "moon_distance_km": dist_moon_m / 1000.0,
            "moon_px": moon_px,
            "sun_px": sun_px,
            "scale_pct": (moon_px / 500.0) * 100.0,
            "screen_x_px": sep_x_px,
            "screen_y_px": sep_y_px,
            "alpha_deg": math.degrees(alpha),
            "beta_deg": math.degrees(beta),
            "angular_sep_deg": math.degrees(ang_sep),
            "sun_az_deg": float(sun_topo.az.to(u.deg).value),
            "sun_alt_deg": float(sun_topo.alt.to(u.deg).value),
            "moon_az_deg": float(moon_topo.az.to(u.deg).value),
            "moon_alt_deg": float(moon_topo.alt.to(u.deg).value)
        }
        rows.append(row)
        sep_offsets.append(math.hypot(sep_x_px, sep_y_px))

    df = pd.DataFrame(rows)
    df.to_csv(out_csv_local, index=False)
    print("Wrote CSV:", out_csv_local)

    # compute center row index: smallest absolute time_s_center
    if "time_s_center" in df.columns:
        center_row_index = int(df["time_s_center"].abs().idxmin())
        center_row_time = float(df.loc[center_row_index, "time_s_center"])
    else:
        center_row_index = int(frames_local // 2)
        center_row_time = float(0.0)

    real_duration_s = 2.0 * half
    time_compression = real_duration_s / (frames_local / fps) if fps > 0 else None
    meta = {
        "center_utc": refined_time_astropy.iso,
        "center_et": float(center_et) if center_et is not None else None,
        "half_window_s": half,
        "real_duration_s": real_duration_s,
        "frames": frames_local,
        "fps": fps,
        "time_compression": time_compression,
        "ae_center_frame": ae_center_frame,
        "center_row_index": center_row_index,
        "center_row_time_s_center": center_row_time
    }
    with open(METADATA_JSON, "w") as fh:
        json.dump(meta, fh, indent=2)
    print("Wrote metadata:", METADATA_JSON)
    print("Max projected offset (sensor px):", float(np.nanmax(sep_offsets)))
    print("Final refined center UTC:", refined_time_astropy.iso)
    if center_et is not None:
        print("Final refined center ET (s):", center_et)

if __name__ == "__main__":
    main()
