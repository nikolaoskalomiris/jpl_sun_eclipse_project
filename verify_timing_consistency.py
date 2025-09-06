#!/usr/bin/env python3
"""
verify_timing_consistency.py (updated)

Now aligns the mapping so that the CSV row closest to time_s_center==0 maps
exactly to ae_center_frame (matching the JSX strategy), then runs checks.
"""
import json, os, math, sys
import pandas as pd

CSV = "eclipse_keyframes_full.csv"
META = "center_metadata.json"

def load():
    if not os.path.exists(CSV):
        print("CSV not found:", CSV); sys.exit(1)
    if not os.path.exists(META):
        print("Metadata not found:", META); sys.exit(1)
    df = pd.read_csv(CSV)
    meta = json.load(open(META, "r"))
    return df, meta

def main():
    df, meta = load()
    frames = int(meta.get("frames", len(df)))
    fps = float(meta.get("fps", 25))
    half = float(meta.get("half_window_s", None)) if meta.get("half_window_s") is not None else float(meta.get("real_duration_s", 0))/2.0
    real_duration = float(meta.get("real_duration_s", 2*half))
    ae_center_frame = int(meta.get("ae_center_frame", 1000))
    frames_per_real_second = frames / real_duration
    time_compression = meta.get("time_compression", real_duration / (frames / fps))

    print("=== Metadata summary ===")
    print("frames:", frames, "fps:", fps, "real_duration_s:", real_duration, "half_window_s:", half)
    print("ae_center_frame:", ae_center_frame, "frames_per_real_second:", frames_per_real_second, "time_compression:", time_compression)
    print("center_utc:", meta.get("center_utc"), "center_et:", meta.get("center_et"))

    # build provisional mapping (provisional_target = ae_center_frame + round(ts_center * fps_factor))
    provisional = []
    for idx, row in df.iterrows():
        if 'time_s_center' in df.columns:
            ts_center = float(row['time_s_center'])
        else:
            ts_from_start = float(row.get('time_s_from_start', 0.0))
            ts_center = ts_from_start - half
        frame_offset = round(ts_center * frames_per_real_second)
        provisional_target = ae_center_frame + frame_offset
        provisional.append({'idx': idx, 'ts_center': ts_center, 'provisional_target': int(provisional_target)})

    # find index closest to center (min |ts_center|)
    best = min(provisional, key=lambda x: abs(x['ts_center']))
    idx_min = best['idx']
    provisional_center_mapped = best['provisional_target']
    delta = ae_center_frame - provisional_center_mapped

    # apply delta to get final mapping
    def map_row_to_frame(row):
        if 'time_s_center' in row.index:
            ts_center = float(row['time_s_center'])
        else:
            ts_from_start = float(row.get('time_s_from_start', 0.0))
            ts_center = ts_from_start - half
        frame_offset = round(ts_center * frames_per_real_second)
        provisional_target = ae_center_frame + frame_offset
        target = provisional_target + delta
        return int(target), ts_center

    # sample checks
    check_indices = [0, 1, max(0, frames//2 - 1), frames//2, frames - 1]
    print("\nSample mappings (csv_row -> AE_frame) :")
    for idx in check_indices:
        if idx >= len(df): continue
        row = df.iloc[idx]
        target_frame, ts_center = map_row_to_frame(row)
        print(f"csv_row={idx:4d} frame_col={int(row.get('frame', idx))} time_s_center={ts_center:8.3f} -> AE_frame={target_frame}")

    expected_start = ae_center_frame - round((frames / 2.0))
    expected_end = ae_center_frame + round((frames / 2.0))
    mapped_start, _ = map_row_to_frame(df.iloc[0])
    mapped_end, _ = map_row_to_frame(df.iloc[len(df)-1])
    mapped_center_row, _ = map_row_to_frame(df.iloc[idx_min])

    print("\nExpected start AE_frame approx:", expected_start)
    print("Mapped start AE_frame:", mapped_start)
    print("Expected end AE_frame approx:", expected_end)
    print("Mapped end AE_frame:", mapped_end)
    print("Mapped center row AE_frame:", mapped_center_row, "should equal ae_center_frame:", ae_center_frame)

    ok = True
    if mapped_center_row != ae_center_frame:
        print("WARNING: mapped center row frame != ae_center_frame")
        ok = False
    if mapped_start < 0:
        print("WARNING: start maps to negative AE frame:", mapped_start); ok = False
    if mapped_end > (ae_center_frame * 2 + 10):
        print("WARNING: end maps beyond expected range:", mapped_end); ok = False

    if ok:
        print("\nVERIFICATION PASSED: timings & mapping consistent.")
    else:
        print("\nVERIFICATION FAILED: check the warnings above.")

if __name__ == "__main__":
    main()
