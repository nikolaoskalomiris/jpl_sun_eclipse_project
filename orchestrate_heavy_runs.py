#!/usr/bin/env python3
"""
orchestrate_heavy_runs.py (patched - robust extra-args, resume, logging, safe merge)

Features:
  - Accepts --extra-args as argparse.REMAINDER so you don't have to quote flags.
  - Skips frames already present in out_root (resume behavior).
  - Writes per-frame worker stdout/stderr into out_root/logs/frame_XXXXX.log
  - Does not delete any per-frame CSVs.
  - Merges any existing per-frame CSVs found in out_root at the end.
  - Still uses multiprocessing with top-level run_task (pickleable).

Usage example:
 python orchestrate_heavy_runs.py \
   --ae-csv eclipse_keyframes_full.csv \
   --candidates candidate_frames.csv \
   --heavy-script generate_moon_limb_profile_with_occlusion.py \
   --out-root out_runs \
   --concurrency 6 \
   --chunk-size 20 \
   --extra-args --preview-n-angles 256
"""
from __future__ import annotations
import os
import sys
import argparse
import subprocess
import json
import time
import multiprocessing as mp
from typing import List, Tuple, Any, Optional
import pandas as pd
from astropy.time import Time, TimeDelta

# ---------------- helper functions (top-level so picklable) ----------------

def read_metadata(meta_path: str):
    if not os.path.exists(meta_path):
        return None
    with open(meta_path, "r") as fh:
        return json.load(fh)

def frame_to_utc_for_ae_row(row: dict, meta: dict) -> str:
    """
    Given an AE CSV row dict (must contain time_s or time_s_center/time_s_from_start),
    and metadata with center_utc, return an ISO UTC string for the frame.
    """
    if meta is None:
        raise RuntimeError("Missing center metadata JSON; cannot map frames to UTC.")
    center_utc = meta.get("center_utc")
    if center_utc is None:
        raise RuntimeError("Metadata missing 'center_utc'")
    # prefer explicit time_s
    if 'time_s' in row and not pd.isna(row['time_s']):
        ts = float(row['time_s'])
    elif 'time_s_center' in row and not pd.isna(row['time_s_center']):
        ts = float(row['time_s_center'])
    elif 'time_s_from_start' in row and not pd.isna(row['time_s_from_start']):
        half = meta.get("half_window_s")
        if half is None:
            half = meta.get("real_duration_s") / 2.0 if meta.get("real_duration_s") else None
        if half is None:
            raise RuntimeError("Cannot compute time_s from time_s_from_start without half_window_s or real_duration_s in metadata.")
        ts = float(row['time_s_from_start']) - float(half)
    else:
        raise RuntimeError("AE row lacks time_s/time_s_center/time_s_from_start")

    center_t = Time(center_utc, scale="utc")
    t = center_t + TimeDelta(ts, format="sec")
    return t.iso

def run_worker_for_frame(heavy_script: str, utc_iso: str, out_csv: str, extra_args: Optional[List[str]] = None) -> Tuple[bool,str]:
    """
    Launch a subprocess that runs the heavy script for a single UTC and writes out_csv.
    Returns (ok, combined_stdout_stderr).
    """
    # Build CLI; heavy script expected to accept --utc and --out-csv
    cmd = [sys.executable, heavy_script, "--utc", utc_iso, "--out-csv", out_csv]
    if extra_args:
        # extra_args is a list of tokens (e.g. ["--preview-n-angles","256"])
        cmd += extra_args
    try:
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
        out = (proc.stdout or "") + ("\n" + (proc.stderr or "")) if (proc.stderr or proc.stdout) else ""
        return True, out
    except subprocess.CalledProcessError as e:
        out = (e.stdout or "") + ("\n" + (e.stderr or ""))
        return False, out
    except Exception as e:
        return False, str(e)

def run_task(task_tuple: Tuple[Any, ...]) -> List[Tuple[int, bool, str, str]]:
    """
    Top-level task runner used by worker processes.
    task_tuple layout:
      (frame_start:int, frame_end:int,
       ae_csv:str, meta_path:str,
       heavy_script:str, out_root:str, extra_args:list_or_none)
    Returns list of (frame_idx, ok_bool, out_csv_path, log_text)
    """
    (frame_start, frame_end, ae_csv, meta_path, heavy_script, out_root, extra_args) = task_tuple
    results = []
    # Load AE CSV locally inside worker (fast; small)
    try:
        ae_df_local = pd.read_csv(ae_csv, engine="python")
    except Exception:
        ae_df_local = pd.read_csv(ae_csv)
    meta = read_metadata(meta_path) if meta_path and os.path.exists(meta_path) else None

    logs_dir = os.path.join(out_root, "logs")
    os.makedirs(logs_dir, exist_ok=True)

    for fi in range(int(frame_start), int(frame_end) + 1):
        # expected output path for this frame
        out_csv = os.path.join(out_root, f"frame_{int(fi):05d}_limb.csv")
        log_path = os.path.join(logs_dir, f"frame_{int(fi):05d}.log")

        # If file already exists (resume), skip running and write a "skipped" log entry
        if os.path.exists(out_csv):
            msg = f"[SKIP] output exists: {out_csv}\n"
            with open(log_path, "a", encoding="utf-8") as L:
                L.write(msg)
            results.append((fi, True, out_csv, msg))
            continue

        # find AE row for frame fi
        row_df = None
        try:
            if 'frame' in ae_df_local.columns:
                row_df = ae_df_local.loc[ae_df_local['frame'] == fi]
        except Exception:
            row_df = None
        if row_df is None or row_df.shape[0] == 0:
            # fallback to positional index if possible
            if fi < 0 or fi >= len(ae_df_local):
                log = f"Frame {fi} out of range for AE CSV (len={len(ae_df_local)})"
                with open(log_path, "a", encoding="utf-8") as L:
                    L.write(log + "\n")
                results.append((fi, False, "", log))
                continue
            row = ae_df_local.iloc[fi].to_dict()
        else:
            row = row_df.iloc[0].to_dict()

        try:
            utc = frame_to_utc_for_ae_row(row, meta)
        except Exception as e:
            log = f"Error mapping frame->UTC: {e}"
            with open(log_path, "a", encoding="utf-8") as L:
                L.write(log + "\n")
            results.append((fi, False, "", log))
            continue

        ok, logtxt = run_worker_for_frame(heavy_script, utc, out_csv, extra_args)
        # always write a per-frame log so user can inspect
        with open(log_path, "a", encoding="utf-8") as L:
            L.write(f"UTC: {utc}\nExitOK: {ok}\n\nSTDOUT+STDERR:\n")
            L.write(logtxt or "")
        # record result; even if ok is False, the worker may still have produced a file
        results.append((fi, ok, out_csv if os.path.exists(out_csv) else "", logtxt or ""))
    return results

# ---------------- main() ----------------

def main(argv=None):
    p = argparse.ArgumentParser(description="orchestrate_heavy_runs.py (patched)")
    p.add_argument("--ae-csv", dest="ae_csv", required=True, help="eclipse_keyframes_full.csv (with time_s_center)")
    p.add_argument("--candidates", dest="candidates", required=True, help="candidate_frames.csv (if missing and --contacts provided, will be generated)")
    p.add_argument("--contacts", dest="contacts", required=False, help="(optional) contacts.csv; used to auto-generate candidates if necessary")
    p.add_argument("--heavy-script", dest="heavy_script", required=True, help="path to generate_moon_limb_profile_with_occlusion.py (CLI)")
    p.add_argument("--out-root", dest="out_root", required=True, help="output root directory for runs")
    p.add_argument("--concurrency", dest="concurrency", type=int, default=max(1, mp.cpu_count()//2), help="max parallel workers")
    p.add_argument("--chunk-size", dest="chunk_size", type=int, default=1, help="frames per worker invocation (1 = one invocation per frame)")
    # Use REMAINDER so the user may type: --extra-args --preview-n-angles 256
    p.add_argument("--extra-args", dest="extra_args", nargs=argparse.REMAINDER,
                   help="Extra args forwarded to heavy script. Example: --extra-args --preview-n-angles 256")
    args = p.parse_args(argv)

    ae_csv = args.ae_csv
    candidates_csv = args.candidates
    contacts_csv = args.contacts
    heavy_script = args.heavy_script
    out_root = args.out_root
    concurrency = max(1, int(args.concurrency))
    chunk_size = max(1, int(args.chunk_size))
    extra_args = args.extra_args if args.extra_args else None
    # If extra_args provided via REMAINDER, it may include a leading '--' token depending on how user typed it.
    # Normalize: if first token is '--', drop it.
    if extra_args and len(extra_args) > 0 and extra_args[0] == "--":
        extra_args = extra_args[1:]
    # If user passed a single quoted string, split it (backwards-compatible)
    if extra_args and isinstance(extra_args, list) and len(extra_args) == 1 and isinstance(extra_args[0], str) and " " in extra_args[0]:
        extra_args = extra_args[0].strip().split()

    if not os.path.exists(ae_csv):
        raise RuntimeError("AE CSV not found: " + ae_csv)
    if not os.path.exists(heavy_script):
        raise RuntimeError("heavy script not found: " + heavy_script)
    os.makedirs(out_root, exist_ok=True)
    logs_dir = os.path.join(out_root, "logs")
    os.makedirs(logs_dir, exist_ok=True)

    # If candidates CSV doesn't exist but user supplied --contacts, try to generate it
    if not os.path.exists(candidates_csv):
        if contacts_csv:
            print(f"Candidates file '{candidates_csv}' not found but --contacts provided. Attempting to generate candidates using select_eclipse_frames_from_ae.py ...")
            sel_script = os.path.join(os.path.dirname(__file__), "select_eclipse_frames_from_ae.py")
            if not os.path.exists(sel_script):
                # try in PATH / current dir
                sel_script = "select_eclipse_frames_from_ae.py"
            cmd = [sys.executable, sel_script, "--ae-csv", ae_csv, "--contacts", contacts_csv, "--out", candidates_csv]
            try:
                proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
                print("select_eclipse_frames_from_ae.py stdout:")
                print(proc.stdout)
                if proc.stderr:
                    print("select_eclipse_frames_from_ae.py stderr:")
                    print(proc.stderr)
            except subprocess.CalledProcessError as e:
                print("Failed to auto-generate candidates CSV with select_eclipse_frames_from_ae.py")
                print("stdout:", e.stdout)
                print("stderr:", e.stderr)
                raise RuntimeError("Could not generate candidates CSV; aborting.")
        else:
            raise RuntimeError("candidates CSV not found and no --contacts provided to generate it.")

    if not os.path.exists(candidates_csv):
        raise RuntimeError("candidates CSV not found: " + candidates_csv)

    # read metadata path (prefer center_metadata.json next to ae_csv)
    meta_path_candidate = os.path.join(os.path.dirname(ae_csv), "center_metadata.json")
    meta_path = meta_path_candidate if os.path.exists(meta_path_candidate) else ("center_metadata.json" if os.path.exists("center_metadata.json") else None)
    if meta_path is None:
        raise RuntimeError("center_metadata.json not found; run generate_eclipse_csv.py first")

    # read candidates and expand into per-frame chunks
    cand_df = pd.read_csv(candidates_csv)
    tasks_raw: List[Tuple[int,int]] = []
    for _, row in cand_df.iterrows():
        if 'frame_start_expanded' in row and not pd.isna(row['frame_start_expanded']):
            start = int(row['frame_start_expanded']); end = int(row['frame_end_expanded'])
        elif 'frame_start' in row and not pd.isna(row['frame_start']):
            start = int(row['frame_start']); end = int(row['frame_end'])
        else:
            continue
        f = start
        while f <= end:
            chunk_end = min(end, f + chunk_size - 1)
            tasks_raw.append((f, chunk_end))
            f = chunk_end + 1

    # build tasks, but skip chunks if all frames already have outputs (resume)
    tasks_for_pool = []
    skipped_chunks = 0
    total_chunks = 0
    for (fs, fe) in tasks_raw:
        total_chunks += 1
        # check whether all outputs in this chunk already exist
        all_exist = True
        for fi in range(fs, fe+1):
            expected = os.path.join(out_root, f"frame_{int(fi):05d}_limb.csv")
            if not os.path.exists(expected):
                all_exist = False
                break
        if all_exist:
            skipped_chunks += 1
            continue
        tasks_for_pool.append((fs, fe, ae_csv, meta_path, heavy_script, out_root, extra_args))

    print(f"Prepared {len(tasks_for_pool)} tasks (chunk_size={chunk_size}) from {total_chunks} candidate chunks. Skipped {skipped_chunks} already-complete chunks. Running with concurrency={concurrency} ...")
    start_total = time.time()

    pool = mp.Pool(processes=min(concurrency, max(1, mp.cpu_count()-1)))
    try:
        mapped = pool.map(run_task, tasks_for_pool)
    finally:
        pool.close()
        pool.join()

    # flatten mapped results
    flat = []
    for item in mapped:
        if isinstance(item, list):
            flat.extend(item)
        else:
            flat.append(item)

    # Write a summary log of failures
    fails = [x for x in flat if not x[1]]
    print(f"Total reported runs (from workers): {len(flat)}  failures (worker exitcode != 0): {len(fails)}")
    if fails:
        print("Sample failures (first 10):")
        for f in fails[:10]:
            fi, ok, outcsv, logtxt = f
            print(f" frame {fi} ok={ok} out={outcsv}")

    # Merge produced per-frame CSVs (preserve frame order)
    # Find all files matching pattern frame_XXXXX_limb.csv in out_root
    all_files = []
    for fname in os.listdir(out_root):
        if fname.startswith("frame_") and fname.endswith("_limb.csv"):
            try:
                frame_idx = int(fname.split("_")[1])
            except Exception:
                # try alternative split
                m = fname.replace("frame_","").split("_")[0]
                try:
                    frame_idx = int(m)
                except Exception:
                    continue
            all_files.append((frame_idx, os.path.join(out_root, fname)))
    all_files_sorted = sorted(all_files, key=lambda x: x[0])

    if not all_files_sorted:
        print("No worker CSVs found to merge in out_root:", out_root)
        print("Checked folder contents:", os.listdir(out_root)[:50])
        print("No files will be deleted. Exiting without merge.")
        sys.exit(0)

    merged_rows = []
    for fi, path in all_files_sorted:
        if not os.path.exists(path):
            print("Warning: expected CSV missing at merge time:", path)
            continue
        try:
            df = pd.read_csv(path)
            # if the worker didn't tag frame source, add it
            if '_frame_source' not in df.columns:
                df['_frame_source'] = fi
            merged_rows.append(df)
        except Exception as e:
            print("Warning: failed to read worker CSV:", path, "->", e)

    if merged_rows:
        merged = pd.concat(merged_rows, ignore_index=True)
        merged_out_path = os.path.join(out_root, "merged_moon_limb_profile.csv")
        merged.to_csv(merged_out_path, index=False)
        print("Wrote merged CSV:", merged_out_path)
    else:
        print("No valid per-frame CSVs to merge after reading files. Exiting.")

    elapsed_total = time.time() - start_total
    print(f"All tasks finished in {elapsed_total:.1f}s")

if __name__ == "__main__":
    main()
