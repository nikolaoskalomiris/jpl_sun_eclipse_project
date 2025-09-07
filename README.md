[![Python](https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=fff)](#) [![JSON](https://img.shields.io/badge/JSON-000?logo=json&logoColor=fff)](#) [![JavaScript](https://img.shields.io/badge/JavaScript-F7DF1E?logo=javascript&logoColor=000)](#) ![GitHub commit activity](https://img.shields.io/github/commit-activity/t/nikolaoskalomiris/jpl_sun_eclipse_project) ![GitHub contributors](https://img.shields.io/github/contributors/nikolaoskalomiris/jpl_sun_eclipse_project) ![GitHub Issues or Pull Requests](https://img.shields.io/github/issues/nikolaoskalomiris/jpl_eclipse_project)

**<ins>Abstract:</ins>**

The concept of the project is to create a simulation of the Total Solar Eclipse event taking place at 29th of March 2006 as if it is being observed and recorded from Earth. Along with various places on Earth that day, Kastelorizo, which is a small island located in Greece, was one of the places inside the path of totality. According to NASA documentation the approximate center of totality was around `10:54:04am UTC.` We will gather all the necessary data of the path that each celestial body takes, and import each specific frame in After Effects with a JSX importer. Every other aspect of the bodies will be designed with `Red Giant Trapcode Particular` and `Red Giant Geo.` (Described below)

**<ins>We will split the procedure in parts:</ins>**

- Shape & surface of the Moon
- Baily's Beads & Diamond Ring
- Shape & surface of the Sun
- Sun Explosions
- CMEs
- Corona

This repository will cover the creation of the Baily's Beads and the Diamond Ring, as well as the direction/angle/size with the help of AstroPy and SPICE tools.

**<ins>Tools we will need:</ins>**

- Leap seconds needed for UTC to ET conversion
- JPL planetary ephemeris
- Planetary constants kernel
- Gravitational constants

**<ins>Quick usage guide (recommended sequence)</ins>**

- Ensure `generate_moon_limb_profile_with_occlusion.py` and `orchestrate_heavy_runs.py` are saved in your project folder and executable with your Python environment. Ensure `naif0012.tls` and other kernels are present in `KERNEL_DIR`.
- Ensure you have `candidate_frames.csv`. Confirm it contains the expanded ranges you want.

**<ins>Step 1 — Profile & get chunk-size suggestion</ins>**

Run this first. It does a short preview run and returns a suggested `--chunk-size` tuned to your machine.

```bash

python orchestrate_heavy_runs.py \
  --ae-csv eclipse_keyframes_full.csv \
  --candidates candidate_frames.csv \
  --heavy-script generate_moon_limb_profile_with_occlusion.py \
  --out-root out_runs \
  --concurrency 3 \
  --chunk-size 8 \
  --auto-chunk --apply-chunk-suggestion \
  --profile-sample-frames 3 \
  --profile-extra-args "--preview-n-angles 128" \
  --extra-args "--no-multiproc" \
  --kernel-dir spice_kernels \
  --dem-path moon_dem/GLD100.tif \
  --verbose
```

When this finishes it will:

- print `seconds_per_frame` (profiling)
- print a suggested `--chunk-size` (and apply it because we passed `--apply-chunk-suggestion`)

**<ins>Step 2 — Final high-quality run (use suggested chunk-size S)</ins>**

Replace S with the suggested chunk-size printed after Step 1. Use slightly aggressive quality flags for production:

```bash

python orchestrate_heavy_runs.py \
  --ae-csv eclipse_keyframes_full.csv \
  --candidates candidate_frames.csv \
  --heavy-script generate_moon_limb_profile_with_occlusion.py \
  --out-root out_runs \
  --concurrency 3 \
  --chunk-size S \
  --extra-args "--n-angles 2048 --ray-step-km 0.20 --coarse-factor 6 --no-multiproc" \
  --kernel-dir spice_kernels \
  --dem-path moon_dem/GLD100.tif \
  --verbose
```


**<ins>Top flags</ins>**

- --ae-csv [filename]
- --candidates [candidate_frames.csv]
- --heavy-script [path] — path to heavy script (usually generate_moon_limb_profile_with_occlusion.py)
- --out-root [dir] — where per-frame CSVs & merged output go (default out_runs)
- --concurrency [n] — how many chunk processes to run in parallel
- --chunk-size [N] — frames per chunk (or let profiling suggest)
- --kernel-dir, --dem-path, --center-metadata — passed to heavy script
- --extra-args "[args]" — extra flags for heavy script (e.g. "--preview-n-angles --n-angles --no-multiproc --ray-step-km --coarse-factor")
- --auto-chunk — run quick profiling to suggest a chunk size
- --apply-chunk-suggestion — automatically apply the suggested chunk size
- --profile-sample-frames [N] — frames used in profiling run (default 1)
- --profile-target-chunk-seconds [sec] — desired seconds per chunk (default 1800)
- --profile-extra-args "[args]" — profile-only extra args
- --profile-read-mb [MB] — DEM micro-benchmark size (used for preview heuristics)
- --no-merge — skip final merge step
- --keep-going — continue despite chunk failures
- --verbose

**<ins>Tuning knobs cheat sheet (effects and costs)</ins>**

- `n_angles ↓ →` linear speedup, but faceting on mask. Good: 64–128 for preview.
- `ray_step_km ↑ →` big speedup; risk: miss small occluders/beads. For preview use 1–2 km.
- `coarse_factor ↓/↑` → affects how coarse the initial sweep is. Lowering reduces false positives but can be slower.
- `--no-multiproc ON` → fewer DEM handles, less I/O; use orchestrator concurrency to control parallelism.
- `--chunk-size` ↑ → huge practical speedup (DEM + SPICE init amortized); use 50–200. Makes each worker compute many frames while loading DEM once.
- Disk on SSD / disable AV → essential. If your DEM file is on a slow drive or scanned by AV per-access, speed will be terrible.

#TO_DO: Expand ReadMe
