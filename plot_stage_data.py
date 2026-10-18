"""Stacked timing bars: offline benchmark runs (time_pipeline .npz) and live logs (nav_node .csv).

usage (laptop now):
    python3 plot_stage_bars.py --laptop <run1.npz> <run2.npz> --laptop-live <live.csv>
usage (after robot day):
    python3 plot_stage_bars.py --laptop <npz...> --laptop-live <csv> --g1 <npz...> --g1-live <csv>

With only .npz runs the bars show the detailed per-stage breakdown. If any live .csv is given,
all bars use three coarse groups (the live log only records these) so they compare directly.
"""
import argparse
import csv
import numpy as np
import matplotlib.pyplot as plt

BUDGET_MS = 100.0

# detailed groups (offline runs only)
FINE = [
    ("Message conversion",         ["convert", "update cloud"]),
    ("Crop & mount correction",    ["crop", "std reorient"]),
    ("Ground plane",               ["ground plane"]),
    ("Obstacle extraction",        ["sections"]),
    ("Occupancy grid & filtering", ["height/occ grids"]),
    ("Rail extraction",            ["extract rails"]),
    ("Snake planning",             ["snake plan"]),
    ("Path following",             ["follower"]),
]
# coarse groups: offline stage names, and the matching live CSV column
COARSE = [
    ("LiDAR Data Extraction",   ["convert"],                                         "convert_ms"),
    ("Perception",           ["update cloud", "crop", "std reorient", "ground plane",
                              "sections", "height/occ grids"],                    "data_process_ms"),
    ("Path Planning & Following", ["extract rails", "snake plan", "follower"],         "snake_plan_ms"),
]


def to_float(s):
    try:
        return float(s)
    except (TypeError, ValueError):
        return np.nan


# ---------- offline runs (.npz) ----------
def load_npz(paths):
    datas, names = [], None
    for p in paths:
        with np.load(p, allow_pickle=False) as f:
            n = [str(x) for x in f["names"]]
            if names is not None and n != names:
                raise ValueError(f"{p}: stage names differ from the first run")
            names = n
            datas.append(f["data"])
    return np.vstack(datas), names


def npz_bar(paths, coarse):
    data, names = load_npz(paths)
    col = {n: k for k, n in enumerate(names)}
    groups = [(g, s) for g, s, _ in COARSE] if coarse else FINE
    means = [data[:, [col[x] for x in stages if x in col]].sum(axis=1).mean() for _, stages in groups]
    total = data[:, [k for k, n in enumerate(names) if n != "read"]].sum(axis=1)
    return means, total


# ---------- live logs (.csv) ----------
def csv_bar(paths):
    rows = []
    for p in paths:
        with open(p, newline="") as f:
            rows += list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"no data rows in {paths}")
    col = lambda k: np.array([to_float(r.get(k)) for r in rows])
    means = [np.nanmean(col(c)) for _, _, c in COARSE]
    total = col("callback_ms")
    return means, total[~np.isnan(total)]


def report(label, total):
    print(f"{label:<24} {len(total):5d} frames | mean {total.mean():6.1f} | median {np.median(total):6.1f} | "
          f"p95 {np.percentile(total, 95):6.1f} | max {total.max():6.1f} ms | "
          f"over {BUDGET_MS:.0f} ms: {100 * np.mean(total > BUDGET_MS):.2f}%")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--laptop", nargs="+", help="offline time_pipeline runs (.npz), laptop")
    ap.add_argument("--laptop-live", nargs="+", help="live nav_node logs (.csv), laptop bag playback")
    ap.add_argument("--g1", nargs="+", help="offline time_pipeline runs (.npz), G1")
    ap.add_argument("--g1-live", nargs="+", help="live nav_node logs (.csv), G1")
    ap.add_argument("--out", default="fig_stage_timing.png")
    args = ap.parse_args()

    coarse = bool(args.laptop_live or args.g1_live)
    entries = [("Laptop\n(offline)", args.laptop, "npz"), ("Laptop\n(live playback)", args.laptop_live, "csv"),
               ("G1\n(offline)", args.g1, "npz"), ("G1\n(live)", args.g1_live, "csv")]

    bars = []
    for label, paths, kind in entries:
        if not paths:
            continue
        means, total = npz_bar(paths, coarse) if kind == "npz" else csv_bar(paths)
        report(label.replace("\n", " "), total)
        bars.append((label, means))
    if not bars:
        ap.error("give at least one of --laptop, --laptop-live, --g1, --g1-live")

    group_names = [g for g, _, _ in COARSE] if coarse else [g for g, _ in FINE]
    fig, ax = plt.subplots(figsize=(4.5 + 1.3 * len(bars), 4.5))
    colors = plt.cm.tab10(np.arange(len(group_names)))
    x = np.arange(len(bars))
    bottoms = np.zeros(len(bars))
    for g, gname in enumerate(group_names):
        vals = np.array([b[1][g] for b in bars])
        ax.bar(x, vals, bottom=bottoms, width=0.5, color=colors[g], label=gname, edgecolor="white", linewidth=0.5)
        bottoms += vals
    for xi, tot in zip(x, bottoms):
        ax.text(xi, tot + 1.5, f"{tot:.1f} ms", ha="center", va="bottom", fontsize=9)

    ax.axhline(BUDGET_MS, color="k", linestyle="--", linewidth=1)
    ax.text(x[-1] + 0.3, BUDGET_MS + 1.5, "100 ms budget", ha="right", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([b[0] for b in bars], fontsize=9)
    ax.set_ylabel("Mean processing time per frame (ms)")
    ax.set_ylim(0, max(BUDGET_MS, bottoms.max()) * 1.15)
    ax.set_xlim(-0.6, len(bars) - 0.4)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8, frameon=False)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(args.out, dpi=300, bbox_inches="tight")
    print(f"saved {args.out}")
    plt.show()


if __name__ == "__main__":
    main()