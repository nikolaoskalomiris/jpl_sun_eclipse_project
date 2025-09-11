# sample_between_times.py
import subprocess, shlex, sys
from datetime import datetime, timedelta

heavy = "generate_moon_limb_profile_with_occlusion.py"
kernel_dir = "spice_kernels"
dem = "moon_dem/GLD100.tif"
n=11  # number of samples (including endpoints)
preview=False

# MODIFY: set these times from your result
t_start = "2006-03-29 10:52:24.605"
t_end   = "2006-03-29 10:52:30.008"

fmt = "%Y-%m-%d %H:%M:%S.%f"
dt0 = datetime.strptime(t_start, fmt)
dt1 = datetime.strptime(t_end, fmt)
delta = (dt1 - dt0)
for i in range(n):
    frac = i/(n-1)
    t = dt0 + frac*delta
    utc = t.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]   # ms precision
    out = f"boundary_sample_{i:02d}.csv"
    cmd = [
        sys.executable, heavy,
        "--utc", utc,
        "--out-csv", out,
        "--n-angles", "2048",
        "--ray-step-km", "0.2",
        "--kernel-dir", kernel_dir,
        "--dem-path", dem
    ]
    print("RUN:", " ".join(shlex.quote(x) for x in cmd))
    subprocess.run(cmd, check=True)
