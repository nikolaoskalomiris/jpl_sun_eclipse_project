#!/usr/bin/env python3
"""
orchestrate_heavy_runs.py (patched)

Now picks a preview_n_angles automatically for the profiling step based on:
 - CPU count (os.cpu_count())
 - disk throughput micro-benchmark on the supplied DEM file (reads a few MB)

CLI additions (profiling already present previously):
  --auto-chunk
  --apply-chunk-suggestion
  --profile-target-chunk-seconds
  --profile-sample-frames
  --profile-extra-args

The auto-preview heuristics only run if neither --preview-n-angles nor --n-angles
were already provided in extra-args/profile-extra-args. If DEM path is missing or
read fails, a conservative preview value is chosen.
"""

from __future__ import annotations
import argparse
import os
import sys
import math
import time
import json
import subprocess
import shlex
from pathlib import Path
from typing import List, Tuple
import concurrent.futures
import pandas as pd
import shutil
import multiprocessing

# ---------------- defaults ----------------
DEFAULT_AE_CSV = "eclipse_keyframes_full.csv"
DEFAULT_CANDIDATES = "candidate_frames.csv"
DEFAULT_HEAVY_SCRIPT = "generate_moon_limb_profile_with_occlusion.py"
DEFAULT_OUT_ROOT = "out_runs"
DEFAULT_CONCURRENCY = 12
DEFAULT_CHUNK_SIZE = 50
DEFAULT_KERNEL_DIR = "spice_kernels"
DEFAULT_DEM_PATH = "moon_dem/GLD100.tif"
DEFAULT_CENTER_META = "center_metadata.json"

# ---------------- helpers ----------------


def parse_args():
    p = argparse.ArgumentParser(description="Orchestrate heavy runs using frame-range chunks (with profiling + auto-preview heuristics).")
    p.add_argument("--ae-csv", dest="ae_csv", default=DEFAULT_AE_CSV, help="AE CSV mapping frames to times.")
    p.add_argument("--candidates", dest="candidates", default=DEFAULT_CANDIDATES, help="candidate_frames.csv")
    p.add_argument("--heavy-script", dest="heavy_script", default=DEFAULT_HEAVY_SCRIPT, help="path to heavy script")
    p.add_argument("--out-root", dest="out_root", default=DEFAULT_OUT_ROOT, help="output root directory")
    p.add_argument("--concurrency", dest="concurrency", type=int, default=DEFAULT_CONCURRENCY, help="number of concurrent chunk processes")
    p.add_argument("--chunk-size", dest="chunk_size", type=int, default=DEFAULT_CHUNK_SIZE, help="frames per chunk (default or overridden by profiling suggestion)")
    p.add_argument("--kernel-dir", dest="kernel_dir", default=DEFAULT_KERNEL_DIR, help="kernel dir passed to heavy script")
    p.add_argument("--dem-path", dest="dem_path", default=DEFAULT_DEM_PATH, help="DEM path passed to heavy script")
    p.add_argument("--center-metadata", dest="center_metadata", default=DEFAULT_CENTER_META, help="center_metadata.json path")
    p.add_argument("--extra-args", dest="extra_args", default="", help="extra args string passed to heavy script (wrap in quotes)")
    p.add_argument("--require-spice", dest="require_spice", action="store_true", help="pass --require-spice to heavy script")
    p.add_argument("--no-merge", dest="no_merge", action="store_true", help="do not perform final merge step")
    p.add_argument("--keep-going", dest="keep_going", action="store_true", help="continue on chunk failures (report at end)")
    p.add_argument("--verbose", dest="verbose", action="store_true", help="verbose logging")

    # profiling options
    p.add_argument("--auto-chunk", dest="auto_chunk", action="store_true", help="run a short profile of heavy script and suggest a chunk-size")
    p.add_argument("--apply-chunk-suggestion", dest="apply_chunk_suggestion", action="store_true", help="if set, override --chunk-size with suggested chunk-size from profiling")
    p.add_argument("--profile-target-chunk-seconds", dest="profile_target_chunk_seconds", type=float, default=1800.0, help="desired seconds per chunk for suggestion (default 1800 s)")
    p.add_argument("--profile-sample-frames", dest="profile_sample_frames", type=int, default=1, help="how many frames to use for the profiling run (default 1)")
    p.add_argument("--profile-extra-args", dest="profile_extra_args", default="", help="extra-args string specifically for profiling run (optional)")
    p.add_argument("--profile-read-mb", dest="profile_read_mb", type=float, default=4.0, help="how many MB to read from DEM for micro-benchmark (default 4MB)")
    return p.parse_args()


def read_candidates_csv(path: str) -> List[Tuple[int, int]]:
    """
    Read candidate CSV and return a list of (frame_start, frame_end) inclusive ranges.
    Expects columns: at least frame_start, frame_end; handles variety of formats.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Candidates CSV not found: {path}")
    df = pd.read_csv(path, engine="python")
    if "frame_start" in df.columns and "frame_end" in df.columns:
        ranges = []
        for _, r in df.iterrows():
            try:
                a = int(r["frame_start"])
                b = int(r["frame_end"])
                if b < a:
                    a, b = b, a
                ranges.append((a, b))
            except Exception:
                continue
        return ranges
    if df.shape[1] >= 2:
        ranges = []
        for _, row in df.iterrows():
            vals = [v for v in row.values[:2]]
            try:
                a = int(float(vals[0])); b = int(float(vals[1]))
                ranges.append((a, b))
            except Exception:
                continue
        if len(ranges) > 0:
            return ranges
    raise RuntimeError("Could not parse candidate_frames CSV. Expected columns frame_start,frame_end.")


def split_range_into_chunks(start: int, end: int, chunk_size: int) -> List[Tuple[int, int]]:
    """Split inclusive [start,end] into list of inclusive chunk tuples of size chunk_size."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    chunks = []
    cur = start
    while cur <= end:
        cend = min(end, cur + chunk_size - 1)
        chunks.append((cur, cend))
        cur = cend + 1
    return chunks


def prepare_tasks(candidate_ranges: List[Tuple[int, int]], chunk_size: int) -> List[Tuple[int, int]]:
    tasks = []
    for (a, b) in candidate_ranges:
        sub = split_range_into_chunks(a, b, chunk_size)
        tasks.extend(sub)
    return tasks


def all_frames_have_ok(out_root: str, frame_start: int, frame_end: int) -> bool:
    """Return True if every frame in [frame_start,frame_end] has an .ok sentinel file."""
    for f in range(frame_start, frame_end + 1):
        okp = os.path.join(out_root, f"frame_{int(f):05d}_limb.csv.ok")
        if not os.path.exists(okp):
            return False
    return True


def build_heavy_cmd(heavy_script: str,
                    frame_start: int,
                    frame_end: int,
                    ae_csv: str,
                    center_meta: str,
                    out_root: str,
                    kernel_dir: str,
                    dem_path: str,
                    require_spice: bool,
                    extra_args: str) -> List[str]:
    """
    Build subprocess command list to invoke heavy script in frame-range mode.
    """
    cmd = [sys.executable, heavy_script,
           "--frame-start", str(int(frame_start)),
           "--frame-end", str(int(frame_end)),
           "--ae-csv", ae_csv,
           "--center-metadata", center_meta,
           "--out-dir", out_root,
           "--kernel-dir", kernel_dir,
           "--dem-path", dem_path]
    if require_spice:
        cmd.append("--require-spice")
    if extra_args:
        try:
            extra_list = shlex.split(extra_args)
        except Exception:
            extra_list = extra_args.split()
        cmd.extend(extra_list)
    return cmd


def run_task_subprocess(cmd: List[str], task_label: str, verbose: bool=False, timeout: int | None=None) -> Tuple[bool, str]:
    """
    Run command via subprocess.run, return (success_bool, combined_output).
    """
    try:
        if verbose:
            print(f"[RUN] {task_label}: {' '.join(shlex.quote(c) for c in cmd)}")
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=timeout)
        out = proc.stdout or ""
        success = proc.returncode == 0
        if verbose:
            print(f"[DONE] {task_label} rc={proc.returncode}")
        return success, out
    except subprocess.TimeoutExpired as ex:
        return False, f"TimeoutExpired: {ex}"
    except Exception as ex:
        return False, f"Exception: {ex}"


def find_frame_csvs_with_ok(out_root: str) -> List[str]:
    """Return list of frame CSV paths that have matching .ok sentinel (sorted by frame index)."""
    p = Path(out_root)
    if not p.exists():
        return []
    csvs = []
    for f in sorted(p.glob("frame_*_limb.csv")):
        ok = f.with_suffix(f.suffix + ".ok")
        if ok.exists():
            csvs.append(str(f))
    def frame_from_name(name: str) -> int:
        bn = os.path.basename(name)
        try:
            num = int(bn.split("_")[1])
        except Exception:
            try:
                num = int(''.join(ch for ch in bn if ch.isdigit()))
            except Exception:
                num = 0
        return num
    csvs.sort(key=frame_from_name)
    return csvs


def merge_frame_csvs(out_root: str, merged_name: str = "merged_moon_limb_profile.csv", verbose: bool=False) -> int:
    csvs = find_frame_csvs_with_ok(out_root)
    if len(csvs) == 0:
        if verbose:
            print("No worker CSVs found to merge.")
        return 0
    dfs = []
    for c in csvs:
        try:
            df = pd.read_csv(c, engine="python")
            dfs.append(df)
        except Exception as ex:
            print(f"Warning: failed reading {c}: {ex}")
    if len(dfs) == 0:
        if verbose:
            print("No readable CSVs found to merge.")
        return 0
    merged = pd.concat(dfs, ignore_index=True, sort=False)
    outpath = os.path.join(out_root, merged_name)
    merged.to_csv(outpath, index=False)
    if verbose:
        print(f"Merged {len(dfs)} files into {outpath}")
    return len(dfs)


# ---------------- Profiling + preview heuristics ----------------

def contains_preview_or_nangles(extra_args: str) -> bool:
    if not extra_args:
        return False
    toks = shlex.split(extra_args)
    for t in toks:
        if t.startswith("--preview-n-angles") or t.startswith("--n-angles"):
            return True
    return False


def contains_no_multiproc(extra_args: str) -> bool:
    if not extra_args:
        return False
    toks = shlex.split(extra_args)
    return "--no-multiproc" in toks


def estimate_disk_throughput_mb_s(path: str, sample_mb: float = 4.0, timeout_s: float = 5.0, verbose: bool=False) -> float | None:
    """
    Micro-benchmark: open 'path' (DEM), read up to sample_mb megabytes sequentially,
    return throughput in MB/s. Returns None on failure.
    This is a lightweight heuristic; OS caching may affect result, but it's useful for
    rough decisions (SSD vs network/HDD).
    """
    try:
        if not os.path.exists(path):
            if verbose:
                print(f"estimate_disk_throughput: path does not exist: {path}")
            return None
        filesize = os.path.getsize(path)
        to_read = int(min(filesize, max(1, int(sample_mb * 1024 * 1024))))
        start = time.perf_counter()
        with open(path, "rb") as fh:
            # read in chunks to avoid memory pressure
            remaining = to_read
            chunk = 256 * 1024
            while remaining > 0:
                to_take = min(chunk, remaining)
                data = fh.read(to_take)
                if not data:
                    break
                remaining -= len(data)
                # optional early timeout
                if (time.perf_counter() - start) > timeout_s:
                    break
        elapsed = time.perf_counter() - start
        if elapsed <= 0:
            elapsed = 1e-6
        read_bytes = to_read - remaining
        mb_s = (read_bytes / (1024.0 * 1024.0)) / elapsed
        if verbose:
            print(f"Disk micro-benchmark: read {read_bytes} bytes in {elapsed:.3f}s -> {mb_s:.2f} MB/s")
        return float(mb_s)
    except Exception as ex:
        if verbose:
            print(f"estimate_disk_throughput failed: {ex}")
        return None


def pick_preview_n_angles(cpu_count: int, disk_mb_s: float | None, verbose: bool=False) -> int:
    """
    Heuristic mapping cpu_count and disk throughput to a preview n_angles.
    - base 128
    - cpu_factor ~ cpu/8 (clamped 0.25..2.0)
    - disk_factor: low (<20 MB/s) ->0.5, medium(20-80)->1.0, high(80-200)->1.5, very high ->2.0
    clamp final to [16, 512]
    """
    base = 128.0
    cpu_factor = float(cpu_count) / 8.0 if cpu_count is not None else 1.0
    cpu_factor = max(0.25, min(cpu_factor, 2.0))
    if disk_mb_s is None:
        disk_factor = 0.8  # unknown -> conservative
    else:
        if disk_mb_s < 20.0:
            disk_factor = 0.5
        elif disk_mb_s < 80.0:
            disk_factor = 1.0
        elif disk_mb_s < 200.0:
            disk_factor = 1.5
        else:
            disk_factor = 2.0
    preview = int(round(base * cpu_factor * disk_factor))
    preview = max(16, min(preview, 512))
    if verbose:
        print(f"pick_preview_n_angles(): cpu_count={cpu_count}, disk_mb_s={disk_mb_s}, cpu_factor={cpu_factor:.2f}, disk_factor={disk_factor:.2f} -> preview={preview}")
    return preview


def profile_sample_run(sample_frame: int, heavy_script: str, ae_csv: str, center_meta: str,
                       out_root: str, kernel_dir: str, dem_path: str, require_spice: bool,
                       base_extra_args: str, profile_extra_args: str,
                       sample_frames: int, profile_read_mb: float = 4.0, verbose: bool=False) -> Tuple[bool, float]:
    """
    Run a short profiling subprocess on 'sample_frames' frames starting at sample_frame.
    Writes outputs to profile_out_dir under out_root and returns (success, seconds_per_frame).
    The profiling command will automatically add a computed --preview-n-angles if the user
    did not provide one, based on CPU and DEM micro-benchmark.
    """
    profile_out_dir = os.path.join(out_root, "profile_run")
    os.makedirs(profile_out_dir, exist_ok=True)

    # build extra args for profile: allow user profile_extra_args to override
    prof_args = base_extra_args or ""
    if profile_extra_args:
        prof_args = prof_args + " " + profile_extra_args

    # if neither n-angles nor preview provided, pick one using heuristics
    if not contains_preview_or_nangles(prof_args):
        # estimate disk throughput (MB/s) by reading a few MB from DEM
        disk_mb_s = estimate_disk_throughput_mb_s(dem_path, sample_mb=profile_read_mb, verbose=verbose)
        # CPU count
        try:
            cpu_count = int(os.cpu_count() or multiprocessing.cpu_count() or 1)
        except Exception:
            cpu_count = 1
        preview_n = pick_preview_n_angles(cpu_count, disk_mb_s, verbose=verbose)
        prof_args = prof_args + f" --preview-n-angles {preview_n}"
    else:
        if verbose:
            print("Profile args already contain preview/n-angles; not auto-picking preview.")

    # ensure no nested multiproc (fast profile)
    if not contains_no_multiproc(prof_args):
        prof_args = prof_args + " --no-multiproc"

    # frame range for profiling
    fs = sample_frame
    fe = sample_frame + max(0, sample_frames - 1)

    cmd = build_heavy_cmd(heavy_script, fs, fe, ae_csv, center_meta, profile_out_dir, kernel_dir, dem_path, require_spice, prof_args)
    label = f"profile_{fs:05d}_{fe:05d}"
    if verbose:
        print(f"Profiling command: {' '.join(shlex.quote(c) for c in cmd)}")
        print("Profile outputs written to:", profile_out_dir)

    t0 = time.time()
    success, out = run_task_subprocess(cmd, label, verbose=verbose, timeout=60*60)  # 1hr timeout for safety
    t1 = time.time()
    elapsed = t1 - t0
    if verbose:
        snippet = (out[:2000] + "...") if out and len(out) > 2000 else out
        print("Profile output (snippet):\n", snippet)
    if not success:
        print("Profiling run failed. Not applying auto-chunk suggestion.")
        return False, float('nan')

    # compute seconds per frame
    spp = elapsed / float(sample_frames) if sample_frames > 0 else elapsed
    # cleanup profile outputs (frames + .ok) to avoid polluting out_root
    try:
        shutil.rmtree(profile_out_dir)
    except Exception:
        pass
    return True, spp


# Utility to quote components for logs
def shlex_quote_like(s: str) -> str:
    try:
        import shlex as _s
        return _s.quote(s)
    except Exception:
        return '"' + s.replace('"', '\\"') + '"'


# ---------------- main ----------------

def main():
    args = parse_args()

    ae_csv = args.ae_csv
    candidates = args.candidates
    heavy_script = args.heavy_script
    out_root = args.out_root
    concurrency = max(1, int(args.concurrency))
    chunk_size = max(1, int(args.chunk_size))
    kernel_dir = args.kernel_dir
    dem_path = args.dem_path
    center_meta = args.center_metadata
    extra_args = args.extra_args or ""
    require_spice = args.require_spice
    verbose = args.verbose
    keep_going = args.keep_going
    no_merge = args.no_merge

    # profiling options
    auto_chunk = args.auto_chunk
    apply_chunk_suggestion = args.apply_chunk_suggestion
    target_chunk_seconds = float(args.profile_target_chunk_seconds)
    profile_sample_frames = max(1, int(args.profile_sample_frames))
    profile_extra_args = args.profile_extra_args or ""
    profile_read_mb = float(args.profile_read_mb)

    # sanity checks
    if not os.path.exists(ae_csv):
        print("Error: AE CSV not found:", ae_csv); sys.exit(1)
    if not os.path.exists(candidates):
        print("Error: candidates CSV not found:", candidates); sys.exit(1)
    if not os.path.exists(heavy_script):
        print("Error: heavy script not found:", heavy_script); sys.exit(1)

    os.makedirs(out_root, exist_ok=True)

    print("Orchestrator configuration:")
    print(" AE CSV:", ae_csv)
    print(" Candidates:", candidates)
    print(" Heavy script:", heavy_script)
    print(" Out root:", out_root)
    print(" Concurrency:", concurrency)
    print(" Chunk size (requested):", chunk_size)
    print(" Kernel dir:", kernel_dir)
    print(" DEM path:", dem_path)
    print(" Center metadata:", center_meta)
    print(" Extra args:", extra_args)
    print(" Require spice:", require_spice)
    print(" Auto-chunk profiling:", auto_chunk)
    print(" Apply suggestion:", apply_chunk_suggestion)
    print(" Profile target seconds:", target_chunk_seconds)
    print(" Profile sample frames:", profile_sample_frames)
    print(" Profile read MB for disk probe:", profile_read_mb)
    print(" Verbose:", verbose)
    print("")

    ranges = read_candidates_csv(candidates)
    if len(ranges) == 0:
        print("No candidate ranges found."); sys.exit(1)

    # Compute total frames for informational use
    total_frames = 0
    for (a, b) in ranges:
        total_frames += (b - a + 1)

    # If auto-chunk requested, perform short profiling run
    seconds_per_frame = None
    suggested_chunk_size = None
    if auto_chunk:
        # pick a representative sample frame: midpoint of first candidate range
        a0, b0 = ranges[0]
        sample_frame = (a0 + b0) // 2
        print(f"Auto-chunk profiling enabled. Will profile starting at sample frame {sample_frame}.")
        ok, spp = profile_sample_run(sample_frame, heavy_script, ae_csv, center_meta, out_root, kernel_dir, dem_path, require_spice, extra_args, profile_extra_args, profile_sample_frames, profile_read_mb=profile_read_mb, verbose=verbose)
        if ok:
            seconds_per_frame = spp
            suggested_chunk_size = max(1, int(round(target_chunk_seconds / seconds_per_frame)))
            # clamp suggestion to reasonable bounds
            suggested_chunk_size = min(max(1, suggested_chunk_size), max(1, total_frames))
            print(f"Profiling result: {seconds_per_frame:.2f} s/frame (avg over {profile_sample_frames} frames).")
            print(f"Suggested --chunk-size to target ~{target_chunk_seconds}s per chunk: {suggested_chunk_size} frames")
            if apply_chunk_suggestion:
                print("Applying suggestion: overriding --chunk-size with suggested value.")
                chunk_size = suggested_chunk_size
        else:
            print("Profiling run failed; using requested --chunk-size:", chunk_size)

    tasks = prepare_tasks(ranges, chunk_size)
    print(f"Prepared {len(tasks)} tasks (chunk_size={chunk_size}). Running with concurrency={concurrency} ...")
    print(f"Total candidate frames: {total_frames}")

    # If profiling succeeded, compute expected runtime per task and print
    if seconds_per_frame is not None:
        print("\nEstimated per-task runtimes (based on profiling):")
        for (fs, fe) in tasks[:min(20, len(tasks))]:
            frames = fe - fs + 1
            est = frames * seconds_per_frame
            print(f"  task {fs:05d}-{fe:05d}: frames={frames} est_time_s={est:.1f} ({est/60.0:.1f} min)")
        if len(tasks) > 20:
            print("  ... (listing first 20 tasks) ...")
        print("")

    start_all = time.time()
    successes = 0
    failures = 0
    task_results = []

    # Use ThreadPoolExecutor to run subprocess workers (safe on Windows).
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as ex:
        future_to_task = {}
        for (fs, fe) in tasks:
            if all_frames_have_ok(out_root, fs, fe):
                if verbose:
                    print(f"Skipping chunk {fs}-{fe}: all frames already done (found .ok)")
                successes += 1
                continue
            cmd = build_heavy_cmd(heavy_script, fs, fe, ae_csv, center_meta, out_root, kernel_dir, dem_path, require_spice, extra_args)
            label = f"chunk_{fs:05d}_{fe:05d}"
            fut = ex.submit(run_task_subprocess, cmd, label, verbose)
            future_to_task[fut] = (fs, fe, cmd, label)

        for fut in concurrent.futures.as_completed(future_to_task):
            fs, fe, cmd, label = future_to_task[fut]
            try:
                ok, out = fut.result()
            except Exception as ex:
                ok = False
                out = f"Exception executing task: {ex}"
            logp = os.path.join(out_root, f"{label}.log")
            try:
                with open(logp, "w", encoding="utf-8") as fh:
                    fh.write("COMMAND:\n")
                    fh.write(" ".join(shlex_quote_like(c) for c in cmd) + "\n\n")
                    fh.write("OUTPUT:\n")
                    fh.write(out or "")
            except Exception:
                pass

            if ok:
                successes += 1
                if verbose:
                    print(f"[OK] {label}")
            else:
                failures += 1
                print(f"[FAIL] {label} (see {logp})")
                if verbose:
                    print(out)
                if not keep_going:
                    print("Aborting due to failure (use --keep-going to continue despite failures).")
                    break

    elapsed_all = time.time() - start_all
    print(f"\nTasks completed. Successes: {successes} Failures: {failures}. Elapsed: {elapsed_all:.1f}s")

    # merge outputs if requested
    if not no_merge:
        merged_count = merge_frame_csvs(out_root, merged_name="merged_moon_limb_profile.csv", verbose=verbose)
        if merged_count == 0:
            print("No worker CSVs found to merge.")
        else:
            print(f"Merged {merged_count} per-frame CSVs into {os.path.join(out_root,'merged_moon_limb_profile.csv')}")

    # print summary of missing frames (if any)
    csvs = find_frame_csvs_with_ok(out_root)
    done_frames = set()
    for c in csvs:
        bn = os.path.basename(c)
        try:
            fnum = int(bn.split("_")[1])
            done_frames.add(fnum)
        except Exception:
            pass

    all_candidate_frames = set()
    for (a, b) in ranges:
        for f in range(a, b + 1):
            all_candidate_frames.add(f)

    missing = sorted(list(all_candidate_frames - done_frames))
    if len(missing) > 0:
        print(f"Missing {len(missing)} frames from candidate ranges. Example missing frames: {missing[:10]}")
    else:
        print("All candidate frames have per-frame CSVs with .ok sentinels.")

    if failures > 0:
        sys.exit(2)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()
