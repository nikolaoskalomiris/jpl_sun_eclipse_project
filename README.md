[![Python](https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=fff)](#) [![JSON](https://img.shields.io/badge/JSON-000?logo=json&logoColor=fff)](#) [![JavaScript](https://img.shields.io/badge/JavaScript-F7DF1E?logo=javascript&logoColor=000)](#) ![GitHub commit activity](https://img.shields.io/github/commit-activity/t/nikolaoskalomiris/jpl_sun_eclipse_project) ![GitHub contributors](https://img.shields.io/github/contributors/nikolaoskalomiris/jpl_sun_eclipse_project) ![GitHub Issues or Pull Requests](https://img.shields.io/github/issues/nikolaoskalomiris/jpl_eclipse_project)



Quick usage guide (recommended sequence)
========================================

- Ensure generate_moon_limb_profile_with_occlusion.py (patched) and orchestrate_heavy_runs.py are saved in your project folder and executable with your Python environment. Ensure naif0012.tls and other kernels are present in KERNEL_DIR.
- Ensure you have candidate_frames.csv. Confirm it contains the expanded ranges you want.
- Run the orchestration (example — preview with fewer samples):

python orchestrate_heavy_runs.py --ae-csv eclipse_keyframes_full.csv --candidates candidate_frames.csv --heavy-script generate_moon_limb_profile_with_occlusion.py --out-root out_runs --concurrency 6 --chunk-size 20 --extra-args --preview-n-angles 256


This will:
==========
- spawn up to 6 parallel child processes,
- run the heavy script for each frame in the candidate expanded ranges (125..1859 plus 982..1016 frames),
- each child writes its CSV into out_runs/frame_XXXX/moon_limb_profile.csv,
- at the end the wrapper writes out_runs/merged_moon_limb_profile.csv combining them.


Notes, caveats & tips
=====================

If you want final quality, run without --preview-n-angles and --chunk-size (or set --preview-n-angles 2048 and --chunk-size 1) Both will falback to the final hi-res defaults if not present.

The patched heavy script still loads SPICE and the DEM per-run. Running many frames in parallel is memory intensive if you run many workers simultaneously. Start with --concurrency ≈ CPU cores − 1 (or lower if you hit memory pressure).

If you prefer the heavy script to reuse a memory-mapped DEM across frames (faster), it is advised to add a server-worker mode later — but the current per-process approach is simpler and robust.

The orchestration merges per-frame CSVs into one file that includes a frame column. You can later filter or summarize as you like.


Tuning knobs cheat sheet (effects and costs)
============================================
- n_angles ↓ → linear speedup, but faceting on mask. Good: 64–128 for preview.
- ray_step_km ↑ → big speedup; risk: miss small occluders/beads. For preview use 1–2 km.
- coarse_factor ↓/↑ → affects how coarse the initial sweep is. Lowering reduces false positives but can be slower.
- --no-multiproc ON → fewer DEM handles, less I/O; use orchestrator concurrency to control parallelism.
- --chunk-size ↑ → huge practical speedup (DEM + SPICE init amortized); use 50–200. Makes each worker compute many frames while loading DEM once.
- Disk on SSD / disable AV → essential. If your DEM file is on a slow drive or scanned by AV per-access, speed will be terrible.
