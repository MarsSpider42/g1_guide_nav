"""Summarise a live timing CSV written by nav_node.py's TimingLogger."""
import csv
import sys
import numpy as np


def to_float(s):
    try:
        return float(s)
    except (TypeError, ValueError):
        return np.nan


def main(path):
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    print(f"file: {path}")
    print(f"rows read: {len(rows)}")
    if not rows:
        print("no data rows: the logger only wrote the header. Was the node stopped with Ctrl+C "
              "(so close() ran), and did any scans arrive? Check the node printed 'logging timing to ...'.")
        return

    col = lambda k: np.array([to_float(r.get(k)) for r in rows])
    status = np.array([r.get("status", "") for r in rows])
    cb, gap = col("callback_ms"), col("stamp_gap_ms")
    ok = status == "ok"

    print(f"{np.sum(~ok)} frames not ok (stops or failures)")
    if np.all(np.isnan(cb)):
        print("callback_ms is empty in every row: check nav_node.py sets res['timing']['callback_ms']")
        return
    print(f"callback (conversion + pipeline): mean {np.nanmean(cb):.1f}  median {np.nanmedian(cb):.1f}  "
          f"p95 {np.nanpercentile(cb, 95):.1f}  max {np.nanmax(cb):.1f} ms")
    print(f"  conversion {np.nanmean(col('convert_ms')):.1f} | data processing {np.nanmean(col('data_process_ms')):.1f} | "
          f"snake + follower {np.nanmean(col('snake_plan_ms')):.1f} ms (means)")
    print(f"frames over 100 ms: {100 * np.mean(cb[~np.isnan(cb)] > 100):.2f}%")

    g = gap[~np.isnan(gap)]
    if len(g):
        skipped = int(max(np.sum(np.round(g / 100.0) - 1), 0))      # ~200 ms gap = 1 scan skipped
        print(f"LiDAR stamp gaps: median {np.median(g):.1f} ms | scans skipped ~{skipped} "
              f"of {len(rows) + skipped} ({100 * skipped / (len(rows) + skipped):.2f}%)")
    print(f"debug publish (not part of budget): mean {np.nanmean(col('publish_ms')):.1f} ms")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: python3 analyse_live_timing.py <live_timing_*.csv>")
    else:
        main(sys.argv[1])