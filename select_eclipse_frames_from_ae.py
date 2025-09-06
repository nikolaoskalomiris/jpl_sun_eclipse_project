#!/usr/bin/env python3
"""
select_eclipse_frames_from_ae.py (patched, robust UTC parsing)

- Robustly loads AE CSV produced by generate_eclipse_csv.py (accepts time_s, time_s_center, time_s_from_start)
- Loads center_metadata.json if present (preferred source for center_utc/center_et/ae_center_frame)
- Loads contacts.csv (several common formats supported) and finds named contacts:
    first_contact, second_contact, center, third_contact, fourth_contact
- Maps contact UTC times to AE frames using metadata (frames_per_real_second) and AE center/frame mapping
- Emits candidate_frames.csv with columns:
    phase,frame_start,frame_end,frame_start_expanded,frame_end_expanded,utc_start,utc_end
- Accepts contact UTC formats like "2006 MAR 29 10:54:04.555", "2006-03-29 10:54:04", iso strings, or ET values.
"""
from __future__ import annotations
import os, sys, argparse, json, math, re
from typing import Optional, Dict, Any
import pandas as pd
from astropy.time import Time, TimeDelta
import datetime

# optional spice for ET/UTC conversions
try:
    import spiceypy as sp
except Exception:
    sp = None

# optional dateutil parser (nice to have)
try:
    from dateutil.parser import parse as dateutil_parse
except Exception:
    dateutil_parse = None

# ---------------- Helpers ----------------

def read_center_metadata(path: str = "center_metadata.json") -> Optional[Dict[str, Any]]:
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r") as fh:
            return json.load(fh)
    except Exception as e:
        print("Warning: failed to read center metadata:", e)
        return None

def load_ae_csv(path: str, meta_path: str = "center_metadata.json") -> pd.DataFrame:
    """
    Load AE CSV and ensure there's a 'time_s' column (seconds from center).
    Accepts CSVs that contain:
      - time_s  (preferred),
      - time_s_center, or
      - time_s_from_start (requires center_metadata.json or will infer center).
    Returns pandas.DataFrame with a guaranteed 'time_s' column.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    # try common read encodings; assume comma-separated
    try:
        df = pd.read_csv(path, engine="python")
    except Exception:
        df = pd.read_csv(path, sep=None, engine="python")
    if 'time_s' in df.columns:
        return df
    if 'time_s_center' in df.columns:
        df['time_s'] = df['time_s_center'].astype(float)
        print("load_ae_csv: constructed 'time_s' from 'time_s_center'")
        return df
    if 'time_s_from_start' in df.columns:
        meta = None
        if os.path.exists(meta_path):
            try:
                meta = json.load(open(meta_path, "r"))
            except Exception:
                meta = None
        tsfs = df['time_s_from_start'].astype(float)
        half = None
        if meta:
            if 'half_window_s' in meta and meta['half_window_s'] is not None:
                half = float(meta['half_window_s'])
            elif 'real_duration_s' in meta and meta['real_duration_s'] is not None:
                half = float(meta['real_duration_s']) / 2.0
        if half is not None:
            df['time_s'] = tsfs - half
            print("load_ae_csv: constructed 'time_s' from 'time_s_from_start' using center_metadata.json")
            return df
        else:
            # infer midpoint
            first = float(tsfs.iloc[0])
            last = float(tsfs.iloc[-1])
            center = 0.5 * (first + last)
            df['time_s'] = tsfs - center
            print("load_ae_csv: constructed 'time_s' from 'time_s_from_start' by inferring center=(first+last)/2")
            return df
    raise RuntimeError(f"AE CSV missing required column 'time_s' and no alternative columns found in {path}")

def try_read_contacts(path: str) -> pd.DataFrame:
    """
    Try flexible parsing of contacts.csv; return a DataFrame with rows indexed by contact name
    If parsing fails, raise.
    Accepts:
     - whitespace or comma separated
     - file with contact as index (first column)
     - file with column 'contact' or 'name'
    """
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    # try several read strategies
    tried = []
    parses = []
    # 1) try with sep regex that handles whitespace or commas
    try:
        df = pd.read_csv(path, sep=r"\s+|,", engine="python", comment="#")
        parses.append(("sep_regex", df))
    except Exception as e:
        tried.append(("sep_regex", str(e)))
    # 2) try simple csv
    try:
        df2 = pd.read_csv(path, engine="python")
        parses.append(("csv", df2))
    except Exception as e:
        tried.append(("csv", str(e)))
    # 3) try whitespace-only
    try:
        df3 = pd.read_csv(path, sep=r"\s+", engine="python", comment="#")
        parses.append(("ws", df3))
    except Exception as e:
        tried.append(("ws", str(e)))
    if not parses:
        raise RuntimeError(f"Failed to parse contacts.csv. Tried: {tried}")
    # pick the parse that has the most useful columns (utc-like or et-like)
    best = None
    best_score = -1
    for name, df in parses:
        cols = set([c.lower() for c in df.columns])
        score = 0
        if 'utc_iso' in cols or 'utc' in cols or 'utc_iso' in cols:
            score += 3
        if 'et' in cols:
            score += 2
        # presence of 'contact' or 'name' column boosts
        if 'contact' in cols or 'name' in cols:
            score += 2
        # try to detect if first column is contact index-like (strings)
        try:
            first_col = df.columns[0]
            if df[first_col].dtype == object:
                score += 1
        except Exception:
            pass
        if score > best_score:
            best_score = score
            best = df.copy()
    df = best
    # Normalize: if there is a 'contact' column, set it as index; else if first column has names, use it
    lc = [c.lower() for c in df.columns]
    if 'contact' in lc:
        # find real column name
        contact_col = df.columns[lc.index('contact')]
        df = df.set_index(contact_col)
    else:
        first_col = df.columns[0]
        # if first column values look like contact names (strings without numbers), set as index
        if df[first_col].dtype == object:
            # but only if many values are non-numeric
            nonnum = df[first_col].apply(lambda x: isinstance(x, str) and not any(ch.isdigit() for ch in x))
            if nonnum.sum() > 0:
                df = df.set_index(first_col)
    # trim whitespace from index
    df.index = df.index.astype(str).str.strip()
    return df

def find_contact_row(df_contacts: pd.DataFrame, name_candidates):
    """
    Search contacts DataFrame index or columns for first matching candidate name.
    name_candidates: list of strings to try in order.
    Returns dict(row) or None.
    """
    for n in name_candidates:
        if n in df_contacts.index:
            row = df_contacts.loc[n]
            # if multiple rows with same index, take first
            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]
            return row.to_dict()
    # try lowercase fuzzy search in index
    idx_lower = {str(i).lower(): i for i in df_contacts.index}
    for n in name_candidates:
        if n.lower() in idx_lower:
            row = df_contacts.loc[idx_lower[n.lower()]]
            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]
            return row.to_dict()
    # try search in 'contact' column if present
    for col in df_contacts.columns:
        if col.lower() in ('contact', 'name'):
            s = df_contacts[col].astype(str).str.strip().str.lower()
            for n in name_candidates:
                mask = s == n.lower()
                if mask.any():
                    row = df_contacts.loc[mask].iloc[0]
                    return row.to_dict()
    # try searching any string cell for candidate substring
    for n in name_candidates:
        lower = n.lower()
        for idx, r in df_contacts.iterrows():
            for c in r:
                try:
                    if isinstance(c, str) and c.strip().lower() == lower:
                        return r.to_dict()
                except Exception:
                    continue
    return None

# ------------------ new: normalize UTC strings ------------------

_MONTHS = {
    'JAN': '01','FEB': '02','MAR': '03','APR': '04','MAY': '05','JUN': '06',
    'JUL': '07','AUG': '08','SEP': '09','OCT': '10','NOV': '11','DEC': '12'
}

def normalize_utc_string(s: str) -> Optional[str]:
    """
    Try to normalize a wide variety of contact time strings into a standard ISO-like string
    that astropy.Time can parse, e.g. "2006-03-29 10:54:04.555".

    Handles forms like:
      - "2006 MAR 29 10:54:04.555"
      - "2006 MAR 29 10:54:04"
      - "2006-03-29 10:54:04"
      - "2006/03/29 10:54:04"
      - variants with commas, multiple spaces

    If normalization succeeds, returns an ISO-ish string. Otherwise returns None.
    """
    if s is None:
        return None
    s0 = str(s).strip()
    if s0 == "":
        return None
    # remove surrounding quotes
    s0 = s0.strip(' "\'')
    # collapse multiple whitespace
    s0 = re.sub(r"\s+", " ", s0)
    # remove stray commas
    s0 = s0.replace(",", "")
    # Common pattern: "YYYY MON DD hh:mm:ss(.sss)"
    # detect 3-letter month tokens
    parts = s0.split(" ")
    if len(parts) >= 4:
        # e.g. ['2006','MAR','29','10:54:04.555']
        if parts[1].upper() in _MONTHS:
            year = parts[0]
            mon = _MONTHS[parts[1].upper()]
            day = parts[2]
            rest = " ".join(parts[3:])
            # ensure day has two digits
            try:
                di = int(day)
                day = f"{di:02d}"
            except Exception:
                pass
            cand = f"{year}-{mon}-{day} {rest}"
            s_try = cand
        else:
            s_try = s0
    else:
        s_try = s0
    # Try straightforward astropy parse attempts
    try_formats = [
        s_try,
        s_try.replace("/", "-"),
        s_try.replace("T", " "),
        # sometimes missing separators
        s_try
    ]
    for tstr in try_formats:
        try:
            tt = Time(tstr, scale="utc")
            return tt.iso
        except Exception:
            pass
    # Try parsing with common datetime formats using datetime.strptime
    dt_formats = [
        "%Y-%m-%d %H:%M:%S.%f",
        "%Y-%m-%d %H:%M:%S",
        "%Y %b %d %H:%M:%S.%f",
        "%Y %b %d %H:%M:%S",
        "%Y-%m-%dT%H:%M:%S.%f",
        "%Y-%m-%dT%H:%M:%S"
    ]
    for fmt in dt_formats:
        try:
            d = datetime.datetime.strptime(s_try, fmt)
            t = Time(d, scale="utc")
            return t.iso
        except Exception:
            continue
    # Try dateutil if available
    if dateutil_parse is not None:
        try:
            d = dateutil_parse(s0)
            if d.tzinfo is None:
                # treat as UTC
                d = d.replace(tzinfo=datetime.timezone.utc)
            t = Time(d)
            return t.utc.iso
        except Exception:
            pass
    # last-ditch: try astropy with explicit formats like 'iso' may still fail,
    # try to extract components with regex: year, month (name or num), day, time
    m = re.match(r"^\s*(\d{4})\s+([A-Za-z]{3}|\d{1,2})\s+(\d{1,2})\s+(\d{1,2}:\d{2}:\d{2}(?:\.\d+)?)", s0)
    if m:
        year = m.group(1)
        mon_raw = m.group(2)
        day = m.group(3)
        timestr = m.group(4)
        if mon_raw.isdigit():
            mon = f"{int(mon_raw):02d}"
        else:
            mon = _MONTHS.get(mon_raw[:3].upper(), "01")
        day = f"{int(day):02d}"
        cand = f"{year}-{mon}-{day} {timestr}"
        try:
            tt = Time(cand, scale="utc")
            return tt.iso
        except Exception:
            pass
    # nothing succeeded
    return None

def utc_from_contact_row(row: Dict[str, Any]) -> Optional[str]:
    """
    Try to extract a UTC ISO string from a contact row dict. Accepts columns:
      - 'utc_iso', 'utc', 'utc_iso_str', 'time', 'utc_time', 'time_iso'
    If row contains 'et', will convert ET->UTC using spice if available.
    Returns ISO string (astropy-compatible) or None.
    """
    if row is None:
        return None
    keys = {k.lower(): k for k in row.keys()}
    # prefer explicit utc_iso/utc fields
    for cand in ('utc_iso', 'utc', 'utc_iso_str', 'time_iso', 'time', 'utc_time'):
        if cand in keys:
            raw = row[keys[cand]]
            norm = normalize_utc_string(raw)
            if norm:
                return norm
            # if normalization failed but value looks like numeric ET, handled below
            break
    # try ET numeric
    if 'et' in keys:
        etval = row[keys['et']]
        try:
            etf = float(etval)
            if sp is not None:
                try:
                    return sp.et2utc(etf, "ISOC", 3)
                except Exception:
                    pass
            # fallback: try astropy Time treating ET as seconds since J2000 (less robust)
            try:
                t = Time(etf, format='sec', scale='tdb')
                return t.utc.iso
            except Exception:
                pass
        except Exception:
            pass
    # attempt to find any string-like field that can be normalized
    for k,v in row.items():
        if isinstance(v, str):
            norm = normalize_utc_string(v)
            if norm:
                return norm
    return None

def seconds_between_utc_strings(utc_a: str, utc_b: str) -> float:
    """
    Returns seconds (utc_a - utc_b)
    utc strings accepted by astropy.Time
    """
    ta = Time(utc_a, scale="utc")
    tb = Time(utc_b, scale="utc")
    dt = ta - tb
    return float(dt.to_value('s'))

def map_utc_to_ae_frame(utc_iso: str, meta: Dict[str,Any]) -> int:
    """
    Convert utc_iso to AE frame index using metadata. Metadata expected to contain:
      center_utc (ISO), ae_center_frame (int), frames (int), real_duration_s (float)
    Mapping: frame = ae_center_frame + round( (utc - center_utc) * frames_per_real_second )
    """
    if meta is None:
        raise RuntimeError("map_utc_to_ae_frame: missing metadata")
    if 'center_utc' not in meta:
        raise RuntimeError("map_utc_to_ae_frame: center_utc missing in metadata")
    center_utc = meta['center_utc']
    if 'frames' in meta and 'real_duration_s' in meta and meta['real_duration_s']:
        frames = int(meta['frames'])
        real_duration = float(meta['real_duration_s'])
        frames_per_real_second = frames / real_duration
    elif 'frames' in meta and 'half_window_s' in meta and meta['half_window_s']:
        frames = int(meta['frames'])
        half = float(meta['half_window_s'])
        real_duration = 2.0 * half
        frames_per_real_second = frames / real_duration
    else:
        raise RuntimeError("map_utc_to_ae_frame: insufficient metadata to compute frames_per_real_second")
    ae_center_frame = int(meta.get('ae_center_frame', 1000))
    dt = seconds_between_utc_strings(utc_iso, center_utc)
    offset_frames = round(dt * frames_per_real_second)
    return int(ae_center_frame + offset_frames)

# ---------------- Main ----------------

def main(argv=None):
    p = argparse.ArgumentParser(description="select_eclipse_frames_from_ae.py (patched)")
    p.add_argument("--ae-csv", dest="ae_csv", default="eclipse_keyframes_full.csv", help="AE CSV from generate_eclipse_csv.py")
    p.add_argument("--contacts", dest="contacts", default="contacts.csv", help="Contacts CSV")
    p.add_argument("--meta", dest="meta", default="center_metadata.json", help="center metadata JSON (produced by generate_eclipse_csv.py)")
    p.add_argument("--out", dest="out", default="candidate_frames.csv", help="Output candidate frames CSV")
    args = p.parse_args(argv)

    ae_csv = args.ae_csv
    contacts_csv = args.contacts
    meta_path = args.meta
    out_csv = args.out

    if not os.path.exists(ae_csv):
        print("Error: AE CSV not found:", ae_csv); sys.exit(2)
    ae_df = load_ae_csv(ae_csv, meta_path)

    meta = read_center_metadata(meta_path)
    if meta is None:
        print("Warning: center metadata not found at", meta_path)
    else:
        print("Loaded center metadata: center_utc =", meta.get("center_utc"), "ae_center_frame =", meta.get("ae_center_frame"))

    # attempt to read contacts
    if not os.path.exists(contacts_csv):
        print("Warning: contacts CSV not found:", contacts_csv)
        contacts_df = None
    else:
        try:
            contacts_df = try_read_contacts(contacts_csv)
            print("Parsed contacts CSV; rows:", len(contacts_df))
        except Exception as e:
            print("Warning: failed to parse contacts.csv:", e)
            contacts_df = None

    # Try to fetch named contacts using tolerant search
    names_for = {
        "first_contact": ["first_contact", "first", "ingress", "start"],
        "second_contact": ["second_contact", "second", "C2", "contact2"],
        "center": ["center", "mid", "minimum", "min"],
        "third_contact": ["third_contact", "third", "C3", "contact3"],
        "fourth_contact": ["fourth_contact", "fourth", "egress", "end"]
    }
    found = {}
    if contacts_df is not None:
        for key, candidates in names_for.items():
            r = find_contact_row(contacts_df, candidates)
            if r is not None:
                found[key] = r
    # For diagnostics
    print("Contacts found (names):", list(found.keys()))

    # Try to extract UTC strings for each contact
    contact_utcs = {}
    for k, row in found.items():
        u = utc_from_contact_row(row)
        if u:
            contact_utcs[k] = u
        else:
            # try to build UTC from et if present
            if 'et' in (r := row):
                try:
                    etval = float(r['et'])
                    if sp is not None:
                        contact_utcs[k] = sp.et2utc(etval, 'ISOC', 3)
                    else:
                        # no spice -> try astropy (less reliable)
                        try:
                            t = Time(etval, format='sec', scale='tdb')
                            contact_utcs[k] = t.utc.iso
                        except Exception:
                            contact_utcs[k] = None
                except Exception:
                    contact_utcs[k] = None
            else:
                contact_utcs[k] = None

    # If center not present in contacts, try from metadata
    if 'center' not in contact_utcs or contact_utcs.get('center') is None:
        if meta and meta.get('center_utc'):
            contact_utcs['center'] = meta['center_utc']
            print("Using center_utc from metadata for 'center' contact:", contact_utcs['center'])

    # map contact UTCs to AE frames where possible
    contact_frames = {}
    for k, utc in contact_utcs.items():
        if utc is None:
            contact_frames[k] = {'utc': None, 'ae_frame': None}
            print(f"Contact {k}: UTC unavailable")
            continue
        try:
            f = map_utc_to_ae_frame(utc, meta) if meta else None
            contact_frames[k] = {'utc': utc, 'ae_frame': f}
            print(f"Mapped contact {k} UTC {utc} -> AE_frame {f}")
        except Exception as e:
            print(f"Warning: could not map contact {k} to AE frame: {e}")
            contact_frames[k] = {'utc': utc, 'ae_frame': None}

    # Derive candidate frame ranges:
    rows_for_output = []

    def add_phase_row(phase_name, start_contact, end_contact):
        start_info = contact_frames.get(start_contact)
        end_info = contact_frames.get(end_contact)
        if (start_info is None) or (end_info is None):
            print(f"Not adding phase {phase_name}: missing contacts {start_contact} or {end_contact}")
            return
        fs = start_info.get('ae_frame')
        fe = end_info.get('ae_frame')
        if fs is None or fe is None:
            print(f"Not adding phase {phase_name}: mapping to AE frames failed for contacts")
            return
        # ensure start <= end
        if fs > fe:
            fs, fe = fe, fs
        # expand by 1 frame as safety (you can change)
        fs_exp = max(0, fs - 1)
        fe_exp = fe + 1
        rows_for_output.append({
            "phase": phase_name,
            "frame_start": int(fs),
            "frame_end": int(fe),
            "frame_start_expanded": int(fs_exp),
            "frame_end_expanded": int(fe_exp),
            "utc_start": str(start_info.get('utc')),
            "utc_end": str(end_info.get('utc'))
        })
        print(f"Added phase {phase_name}: frames {fs}..{fe} (expanded {fs_exp}..{fe_exp})")

    add_phase_row("partial_or_total", "first_contact", "fourth_contact")
    add_phase_row("totality", "second_contact", "third_contact")

    # If nothing added, try a heuristic: if AE CSV has angular_sep_deg and center exists,
    # find frames where angular_sep_deg <= some threshold (from contacts metadata if present)
    if not rows_for_output:
        print("No candidate ranges created from contacts. Trying heuristic from AE CSV...")
        if 'angular_sep_deg' in ae_df.columns and meta:
            try:
                min_sep = float(ae_df['angular_sep_deg'].min())
                sep_threshold = min_sep + 0.1
                mask = ae_df['angular_sep_deg'] <= sep_threshold
                if mask.any():
                    idxs = ae_df.index[mask]
                    fs = int(idxs[0])
                    fe = int(idxs[-1])
                    fs_exp = max(0, fs - 2); fe_exp = fe + 2
                    if meta:
                        t0 = Time(meta['center_utc'], scale="utc")
                        utc_start = (t0 + TimeDelta(float(ae_df.loc[fs,'time_s']), format='sec')).iso
                        utc_end = (t0 + TimeDelta(float(ae_df.loc[fe,'time_s']), format='sec')).iso
                    else:
                        utc_start = ""; utc_end = ""
                    rows_for_output.append({
                        "phase": "heuristic_partial",
                        "frame_start": fs, "frame_end": fe,
                        "frame_start_expanded": fs_exp, "frame_end_expanded": fe_exp,
                        "utc_start": utc_start, "utc_end": utc_end
                    })
                    print("Heuristic range added:", fs, fe)
            except Exception as e:
                print("Heuristic failed:", e)

    # finally, if still nothing, attempt to use center +/- some frames (fallback)
    if not rows_for_output:
        print("Fallback: using ae_center_frame +/- 100 frames as candidate range (very rough).")
        if meta and 'ae_center_frame' in meta:
            center_frame = int(meta['ae_center_frame'])
            fs = max(0, center_frame - 100)
            fe = center_frame + 100
            fs_exp = max(0, fs - 1); fe_exp = fe + 1
            utc_start = ""; utc_end = ""
            rows_for_output.append({
                "phase": "fallback_center_window",
                "frame_start": fs, "frame_end": fe,
                "frame_start_expanded": fs_exp, "frame_end_expanded": fe_exp,
                "utc_start": utc_start, "utc_end": utc_end
            })

    # write candidate_frames.csv
    out_df = pd.DataFrame(rows_for_output, columns=[
        "phase", "frame_start", "frame_end", "frame_start_expanded", "frame_end_expanded", "utc_start", "utc_end"
    ])
    out_df.to_csv(out_csv, index=False)
    print("Wrote candidate frames to:", out_csv)
    print("Rows written:", len(out_df))
    print(out_df.to_string(index=False))

if __name__ == "__main__":
    main()
