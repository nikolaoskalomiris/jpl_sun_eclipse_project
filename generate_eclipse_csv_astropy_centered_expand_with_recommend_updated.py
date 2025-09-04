#!/usr/bin/env python3
"""
generate_eclipse_csv_astropy_centered_expand_with_recommend_updated.py

Astropy-based generator:
 - uses JPL ephemeris 'de440'/'de432s' if available (tries in that order)
 - finds time of minimum Sun-Moon apparent separation near start_utc_iso
 - builds a symmetric window centered on that time
 - automatically expands window if edges remain 'touching' the Sun (so frame 0 is not touching)
 - writes CSV suitable for the JSX importer
 - computes a recommended posMultiplier (CSV sensor-px -> comp-px mapping) so POI offsets stay
   within a desired fraction of a composition width

Dependencies:
 pip install astropy numpy pandas jplephem
"""
import os
import numpy as np
import pandas as pd
import math
from astropy.time import Time
from astropy.coordinates import EarthLocation, AltAz, get_sun
import astropy.units as u

# --- Robust ephemeris selection ---
kernel_used = None
_have_jpl = False
try:
    from astropy.coordinates import solar_system_ephemeris
    # allow explicit BSP path override via environment variable
    env_path = os.environ.get('JPL_EPHEMERIS_PATH', None)
    if env_path:
        try:
            solar_system_ephemeris.set(env_path)
            kernel_used = env_path
            _have_jpl = True
            print(f"Using JPL ephemeris from path: {env_path}")
        except Exception as e:
            print("Warning: failed to set ephemeris from JPL_EPHEMERIS_PATH:", env_path, "->", e)

    if kernel_used is None:
        # try preferred kernel names in order
        preferred_kernels = ['de440', 'de432s']
        for kn in preferred_kernels:
            try:
                solar_system_ephemeris.set(kn)
                kernel_used = kn
                _have_jpl = True
                print("Using JPL ephemeris:", kn)
                break
            except Exception:
                # not available; continue
                pass

    if kernel_used is None:
        print("Warning: could not set any preferred JPL ephemeris (de440/de432s). Falling back to astropy default ephemeris.")
except Exception:
    print("Warning: astropy.coordinates.solar_system_ephemeris not available; falling back to astropy default ephemeris.")
    kernel_used = None
    _have_jpl = False

# Try to import get_moon; otherwise fallback to get_body('moon', ...)
_have_get_moon = False
try:
    from astropy.coordinates import get_moon  # preferred
    _have_get_moon = True
except Exception:
    try:
        from astropy.coordinates import get_body
        _have_get_moon = False
    except Exception:
        raise RuntimeError("Neither get_moon nor get_body available from astropy.coordinates. Install a recent astropy.")

# ---------------- USER PARAMETERS ----------------
frames = 2000
initial_total_duration_seconds = 3.0 * 3600.0  # initial window length (3 hours)

# Observer at Kastelorizo Airport (given)
observer_lat_deg = 36.14265853184001
observer_lon_deg = 29.576375086997015
observer_alt_m = 2.0

# Epoch around which to search for minimum separation (UTC).
start_utc_iso = "2006-03-29 10:53:22.600"  # initial guess / center of coarse search
# search window to locate min (we search +/- search_span_seconds around start_utc_iso)
search_span_seconds = initial_total_duration_seconds

# Camera / sensor / asset
sensor_width_mm = 21.44   # sensor width in mm (horizontal / measured)
sensor_pixels = 4096.0
mm_per_pixel = sensor_width_mm / sensor_pixels
f_mm = 35.0
moon_texture_px = 500.0   # measured in AE

# Desired mapping guidance (for posMultiplier recommendation)
# The script will compute recommended posMultiplier such that the maximum
# projected offset of Moon from Sun (in comp px) <= desiredScreenCoverage * comp_width / 2
comp_width_pixels = 4096.0        # set this to your composition width in AE (default 4096)
desiredScreenCoverage = 0.60      # fraction of comp width that the max offset should occupy (0..1)

# Physical diameters
D_moon_km = 3474.8
D_sun_km  = 1391000.0

out_csv = "eclipse_keyframes_full.csv"

# Safety caps for expansion
max_half_window_seconds = 24.0 * 3600.0  # don't expand beyond +/- 24 hours
expand_factor = 1.5                       # multiply half-window by this when expanding
edge_clearance_factor = 0.9               # require edge_sep > edge_clearance_factor*(moon_radius+sun_radius)

# ---------------- computations ----------------
t0 = Time(start_utc_iso, scale="utc")

# 1) Quick coarse search to find approximate minimum separation
search_N = 4001
search_times = t0 + (np.linspace(-search_span_seconds, search_span_seconds, search_N) * u.s)

loc = EarthLocation(lat=observer_lat_deg*u.deg,
                    lon=observer_lon_deg*u.deg,
                    height=observer_alt_m*u.m)

seps = np.empty(search_N, dtype=float)
for i, tt in enumerate(search_times):
    altaz_frame = AltAz(obstime=tt, location=loc)
    sun_topo = get_sun(tt).transform_to(altaz_frame)
    if _have_get_moon:
        moon_topo = get_moon(tt).transform_to(altaz_frame)
    else:
        moon_topo = get_body('moon', tt).transform_to(altaz_frame)
    seps[i] = sun_topo.separation(moon_topo).to(u.rad).value

min_idx = int(np.argmin(seps))
t_center = search_times[min_idx]
min_sep_deg = (seps[min_idx] * u.rad).to(u.deg).value

# Diagnostic print: ephemeris used + center time
if kernel_used:
    print("Ephemeris used for this run:", kernel_used)
else:
    print("Ephemeris used for this run: (astropy default / builtin)")

print("Found coarse minimum separation at:", t_center.iso, "UTC")
print("Coarse minimum angular separation (deg):", min_sep_deg)

# compute apparent angular diameters at center (use these as baseline radii)
altaz_center = AltAz(obstime=t_center, location=loc)
sun_center = get_sun(t_center).transform_to(altaz_center)
if _have_get_moon:
    moon_center = get_moon(t_center).transform_to(altaz_center)
else:
    moon_center = get_body('moon', t_center).transform_to(altaz_center)

# distances (meters)
dist_moon_center_m = float(moon_center.cartesian.norm().to(u.m).value)
dist_sun_center_m = float(sun_center.cartesian.norm().to(u.m).value)

theta_moon_center_rad = 2.0 * math.atan((D_moon_km*1000.0/2.0) / dist_moon_center_m)
theta_sun_center_rad  = 2.0 * math.atan((D_sun_km*1000.0/2.0)  / dist_sun_center_m)

moon_radius_center_deg = (theta_moon_center_rad/2.0) * (180.0/math.pi)
sun_radius_center_deg  = (theta_sun_center_rad/2.0) * (180.0/math.pi)

touch_threshold_deg = (moon_radius_center_deg + sun_radius_center_deg) * edge_clearance_factor

# Start with the user-specified window half-length
half = initial_total_duration_seconds / 2.0

# Expand window while edge separation is <= touch_threshold_deg (or until max_half_window_seconds)
while True:
    t_edge_before = t_center - (half * u.s)
    t_edge_after  = t_center + (half * u.s)

    altaz_before = AltAz(obstime=t_edge_before, location=loc)
    s_before = get_sun(t_edge_before).transform_to(altaz_before)
    if _have_get_moon:
        m_before = get_moon(t_edge_before).transform_to(altaz_before)
    else:
        m_before = get_body('moon', t_edge_before).transform_to(altaz_before)
    sep_before_deg = float(s_before.separation(m_before).to(u.deg).value)

    altaz_after = AltAz(obstime=t_edge_after, location=loc)
    s_after = get_sun(t_edge_after).transform_to(altaz_after)
    if _have_get_moon:
        m_after = get_moon(t_edge_after).transform_to(altaz_after)
    else:
        m_after = get_body('moon', t_edge_after).transform_to(altaz_after)
    sep_after_deg = float(s_after.separation(m_after).to(u.deg).value)

    min_edge_sep = min(sep_before_deg, sep_after_deg)

    print(f"Testing half-window = {half/3600.0:.3f} h -> edge seps (deg): before={sep_before_deg:.6f}, after={sep_after_deg:.6f}; threshold={touch_threshold_deg:.6f}")

    if min_edge_sep > touch_threshold_deg:
        # edges are safely separated
        break

    # need to expand
    if half * expand_factor > max_half_window_seconds:
        print("Reached maximum allowed half-window. Will stop expanding.")
        break

    half *= expand_factor

# Final total duration (may be greater than initial_total_duration_seconds)
total_duration_seconds = 2.0 * half
print(f"Using total_duration_seconds = {total_duration_seconds/3600.0:.6f} hours (half = {half/3600.0:.6f} h)")
print("Edge separations (deg): before=", sep_before_deg, " after=", sep_after_deg, " (min_edge_sep=", min_edge_sep, ")")

# Build the final symmetric times centered on t_center
final_offsets = np.linspace(-half, half, frames)  # seconds
final_times = t_center + (final_offsets * u.s)

# Build CSV rows and track max sensor-pixel offsets
rows = []
sep_sensor_offsets = []  # radial offsets in sensor pixels
sep_x_list = []
sep_y_list = []

for idx, t in enumerate(final_times):
    altaz_frame = AltAz(obstime=t, location=loc)

    sun_topo = get_sun(t).transform_to(altaz_frame)
    if _have_get_moon:
        moon_topo = get_moon(t).transform_to(altaz_frame)
    else:
        moon_topo = get_body('moon', t).transform_to(altaz_frame)

    sun_az_deg = float(sun_topo.az.to(u.deg).value)
    sun_alt_deg = float(sun_topo.alt.to(u.deg).value)
    moon_az_deg = float(moon_topo.az.to(u.deg).value)
    moon_alt_deg = float(moon_topo.alt.to(u.deg).value)

    # cartesian vectors (meters)
    sun_cart = sun_topo.cartesian.xyz.to(u.m).value
    moon_cart = moon_topo.cartesian.xyz.to(u.m).value

    # Build camera basis where +X points to Sun
    fwd = sun_cart
    fwd_norm = np.linalg.norm(fwd)
    if fwd_norm == 0:
        ex = np.array([1.0,0.0,0.0])
        ey = np.array([0.0,1.0,0.0])
        ez = np.array([0.0,0.0,1.0])
    else:
        ex = fwd / fwd_norm
        ref_up = np.array([0.0, 0.0, 1.0])
        right = np.cross(ref_up, ex)
        rn = np.linalg.norm(right)
        if rn < 1e-12:
            ref_up = np.array([0.0, 1.0, 0.0])
            right = np.cross(ref_up, ex)
            rn = np.linalg.norm(right)
            if rn < 1e-12:
                right = np.array([1.0, 0.0, 0.0]); rn = 1.0
        ey = right / rn
        ez = np.cross(ex, ey)

    mdot_x = float(np.dot(moon_cart, ex))
    mdot_y = float(np.dot(moon_cart, ey))
    mdot_z = float(np.dot(moon_cart, ez))

    alpha = math.atan2(mdot_y, mdot_x)
    beta  = math.atan2(mdot_z, mdot_x)
    ang_sep = math.atan2(math.sqrt(mdot_y*mdot_y + mdot_z*mdot_z), mdot_x)

    dist_moon_m = float(np.linalg.norm(moon_cart))
    dist_sun_m  = float(np.linalg.norm(sun_cart))

    theta_moon = 2.0 * math.atan((D_moon_km*1000.0 / 2.0) / dist_moon_m)
    theta_sun  = 2.0 * math.atan((D_sun_km*1000.0  / 2.0) / dist_sun_m)

    moon_image_mm = 2.0 * f_mm * math.tan(theta_moon / 2.0)
    sun_image_mm  = 2.0 * f_mm * math.tan(theta_sun  / 2.0)

    moon_px = moon_image_mm / mm_per_pixel
    sun_px  = sun_image_mm  / mm_per_pixel

    scale_pct = (moon_px / moon_texture_px) * 100.0

    sep_x_px = (f_mm * math.tan(alpha)) / mm_per_pixel
    sep_y_px = (f_mm * math.tan(beta))  / mm_per_pixel

    sep_sensor_offsets.append(math.hypot(sep_x_px, sep_y_px))
    sep_x_list.append(sep_x_px)
    sep_y_list.append(sep_y_px)

    rows.append({
        "frame": int(idx),
        "time_s": float((t - (t_center - half*u.s)).sec),  # time since window start (0..total_duration_seconds)
        "moon_distance_km": dist_moon_m/1000.0,
        "moon_px": moon_px,
        "sun_px": sun_px,
        "scale_pct": scale_pct,
        "screen_x_px": sep_x_px,
        "screen_y_px": sep_y_px,
        "alpha_deg": math.degrees(alpha),
        "beta_deg": math.degrees(beta),
        "angular_sep_deg": math.degrees(ang_sep),
        "sun_az_deg": sun_az_deg,
        "sun_alt_deg": sun_alt_deg,
        "moon_az_deg": moon_az_deg,
        "moon_alt_deg": moon_alt_deg,
        "node": 0,
        "nu_deg": 0.0,
        "omega_deg": 0.0,
        "Omega_deg": 0.0,
        "inclination_deg": 5.145,
        "a_km": 384400.0,
        "eccentricity": 0.0549
    })

df = pd.DataFrame(rows)
df.to_csv(out_csv, index=False)
print(f"Wrote {out_csv} (frames={frames}, total_duration_seconds={total_duration_seconds}s)")
print("Center time (UTC):", t_center.iso)
print("Minimum angular separation (deg):", min_sep_deg)

# --- compute recommended posMultiplier mapping ---
max_sensor_offset_px = float(np.max(sep_sensor_offsets))
max_abs_sep_x = float(np.max(np.abs(sep_x_list)))
max_abs_sep_y = float(np.max(np.abs(sep_y_list)))

print(f"Max radial offset (sensor px) across window: {max_sensor_offset_px:.3f} px")
print(f"Max abs X offset (sensor px): {max_abs_sep_x:.3f} px, Max abs Y offset (sensor px): {max_abs_sep_y:.3f} px")

# baseline sensorToComp (if you want 1 sensor pixel map to comp_width/sensor_pixels comp pixels)
baseline_sensorToComp = comp_width_pixels / sensor_pixels

# decide recommended posMultiplier so that max_sensor_offset_px * posMultiplier <= (desiredScreenCoverage * comp_width_pixels)/2
if max_sensor_offset_px < 1e-9:
    recommended_posMultiplier = baseline_sensorToComp
else:
    allowed_half_comp = (desiredScreenCoverage * comp_width_pixels) / 2.0
    recommended_posMultiplier = allowed_half_comp / max_sensor_offset_px

# clamp to avoid extremely small or huge suggestions
min_pm = baseline_sensorToComp * 0.05
max_pm = baseline_sensorToComp * 10.0
recommended_posMultiplier_clamped = max(min_pm, min(max_pm, recommended_posMultiplier))

print("\n=== Recommendation for CSV->AE mapping ===")
print(f"Composition width used for recommendation: {comp_width_pixels} px")
print(f"baseline_sensorToComp (comp_px per sensor_px) = comp_width / sensor_pixels = {baseline_sensorToComp:.6f}")
print(f"Desired screen coverage fraction = {desiredScreenCoverage:.3f} (0..1)")
print(f"Allowed half-comp px = {allowed_half_comp:.3f} px")
print(f"Computed recommended posMultiplier (unclamped) = {recommended_posMultiplier:.6f}")
print(f"Recommended posMultiplier (clamped to [{min_pm:.6f}, {max_pm:.6f}]) = {recommended_posMultiplier_clamped:.6f}")
print("→ Use this value for `posMultiplier` in your JSX (or set posMultiplier = 0 in JSX and let it auto-calc from sensor_pixels and comp width).")

# Scientific checks summary
print("\n=== Scientific / modelling notes ===")
print("- Using topocentric AltAz from Astropy: parallax and correct observer geometry included.")
print("- Distances taken from astropy cartesian vectors; angular sizes derived via simple geometry (diameter/distance).")
if kernel_used:
    print(f"- Ephemeris: {kernel_used} used for planetary positions.")
else:
    print("- Ephemeris: astropy builtin/default (no JPL BSP used).")
print("- Limitations: no atmospheric refraction correction applied; limb darkening and eclipse penumbra/umbra ground-track are NOT modelled here.")
print("- For VFX this approach is usually sufficient: it matches apparent positions and apparent angular sizes within arc-second-level ephemeris accuracy.")
print("- If you need sub-arcsecond or ground-track accuracy, you should use full NAIF/SPICE tools and site-specific corrections.")

# print a short "how to use" hint
print("\n=== How to use ===")
print("1) Put the generated CSV into the JSX importer you use.")
print("2) In the JSX, either keep posMultiplier=0 (auto) or set posMultiplier to the recommended value above.")
print("3) If Moon/Sun still look out of range, increase comp_width_pixels in this script to match your AE comp width and re-run to get a new recommendation.")
print("4) If you want a world-scale mapping (1 AE px = 1 km), you can also set posMultiplier manually such that 1 AE px corresponds to the chosen km-per-px mapping (but this often produces very large numeric Z/POI values in AE).")

# end
