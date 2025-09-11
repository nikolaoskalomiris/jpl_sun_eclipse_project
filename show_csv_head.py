# show_csv_head.py
import pandas as pd
import sys
path = sys.argv[1] if len(sys.argv) > 1 else "eclipse_keyframes_full.csv"
print("Reading:", path)
# flexible read: try autodetect sep, fallback to comma
try:
    df = pd.read_csv(path, sep=None, engine="python")
except Exception:
    df = pd.read_csv(path)
print("\nColumns:")
for i,c in enumerate(df.columns):
    print(f" {i:02d}: {c!r}")
print("\nFirst 8 rows (transposed for readability):")
print(df.head(8).T.to_string())
