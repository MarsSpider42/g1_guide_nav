"""Path planner check: is the snake path collision-free between vertices?

For each frame, the straight segments between consecutive VALID snake vertices are sampled
and checked against the occupancy grid. Also reports corridor-mode and stop rates, and lists
failing frames (useful for choosing figure examples).

usage:  python3 planner_check.py 7:200:600 9:300:700        (bag_list index:start:stop)
"""
import os
import sys
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
for _sub in ("nav_source", "datasets"):
    _p = os.path.join(_HERE, _sub)
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

from nav_pipeline import NavPipeline
from lidar_scan_storage import ScanStore, scan_to_xyz
from snake_planner import CELL_SIZE

SAMPLE_STEP_M = CELL_SIZE / 2      # sample segments every half cell


def segment_hits(grid, p0, p1):
    """Number of occupied cells (distinct) crossed by the segment p0 -> p1 (metres, x right / y forward)."""
    origin_col = grid.shape[1] // 2
    n = max(2, int(np.ceil(np.hypot(p1[0] - p0[0], p1[1] - p0[1]) / SAMPLE_STEP_M)) + 1)
    hit = set()
    for t in np.linspace(0.0, 1.0, n):
        x = p0[0] + t * (p1[0] - p0[0])
        y = p0[1] + t * (p1[1] - p0[1])
        row = int(round(y / CELL_SIZE))
        col = int(round(origin_col + x / CELL_SIZE))
        if 0 <= row < grid.shape[0] and 0 <= col < grid.shape[1] and grid[row, col] != 0:
            hit.add((row, col))
    return len(hit)


def check_frame(res):
    """Returns (status, corridor_mode, stopped, cells_hit, first_blocked_rail).
    status: None (no path), 'clear', 'adjacent', or 'jump_only'."""
    grid, rails, vertices = res["grid"], res["rails"], res["vertices"]
    stopped = res["cmd"].status != "ok"
    if not rails:
        return None, False, stopped, 0, None
    closed = sum(1 for r in rails if r.left_closed and r.right_closed)
    corridor = closed / len(rails) > 0.40

    idx = [k for k, r in enumerate(rails) if r.valid]
    if len(idx) < 2:
        return None, corridor, stopped, 0, None
    pts = {k: tuple(float(a) for a in rails[k].to_xy(vertices[k])) for k in idx}

    adjacent_cells, jump_hit, first_rail = 0, False, None
    for a, b in zip(idx[:-1], idx[1:]):
        h = segment_hits(grid, pts[a], pts[b])
        if h:
            if b - a == 1:
                adjacent_cells += h
                first_rail = a if first_rail is None else first_rail
            else:
                jump_hit = True
    if adjacent_cells:
        return "adjacent", corridor, stopped, adjacent_cells, first_rail
    return ("jump_only" if jump_hit else "clear"), corridor, stopped, 0, None



def run(bag_path, start, stop):
    store = ScanStore(bag_path)
    stop = len(store) if stop is None else min(stop, len(store))
    pipe = NavPipeline()                                   # same config as the live system
    rows = []
    for i in range(start, stop):
        res = pipe.process(scan_to_xyz(store[i]))
        status, corridor, stopped, cells, first_rail = check_frame(res)
        rows.append((i, status, corridor, stopped, cells, first_rail))
    return rows


def summarise(rows, label):
    with_path = [r for r in rows if r[1] is not None]
    n = max(1, len(with_path))
    count = lambda s: sum(1 for r in with_path if r[1] == s)
    print(f"\n===== {label} =====")
    print(f"frames: {len(rows)} | frames with a snake path: {len(with_path)}")
    print(f"clear: {count('clear')} ({100 * count('clear') / n:.1f}%) | "
          f"blocked only across skipped rails: {count('jump_only')} ({100 * count('jump_only') / n:.1f}%) | "
          f"blocked between adjacent rails: {count('adjacent')} ({100 * count('adjacent') / n:.1f}%)")

    adj = [r for r in with_path if r[1] == "adjacent"]
    if adj:
        cells = np.array([r[4] for r in adj])               # cells_hit
        rails_at = np.array([r[5] for r in adj])            # first blocked rail index
        print("\nadjacent-rail blocks, by number of occupied cells crossed:")
        for lo, hi, name in ((1, 2, "1-2 cells (grazing)"), (3, 5, "3-5 cells"), (6, 10**6, "6+ cells")):
            k = np.sum((cells >= lo) & (cells <= hi))
            print(f"  {name:<20} {k:5d} ({100 * k / len(adj):.1f}%)")
        print("first blocked segment starts at rail (0 = nearest the robot):")
        for r_i in range(int(rails_at.max()) + 1):
            k = np.sum(rails_at == r_i)
            if k:
                print(f"  rail {r_i:2d}: {k}")
        # blocked AND the robot was not stopped: the cases the follower's own check did not catch
        unsafe = [r for r in with_path if r[1] in ("adjacent", "jump_only") and not r[3]]
        unsafe.sort(key=lambda r: r[4], reverse=True)       # most occupied cells crossed first
        print(f"\nblocked but NOT stopped: {len(unsafe)} frames "
              f"({100 * len(unsafe) / n:.1f}% of frames with a path)")
        print(f"{'frame':>6}{'cells':>7}{'rail':>6}  status")
        for r in unsafe:
            rail = "-" if r[5] is None else r[5]
            print(f"{r[0]:>6}{r[4]:>7}{rail:>6}  {r[1]}")

    print(f"\ncorridor mode: {100 * np.mean([r[2] for r in rows]):.1f}% of frames | "
          f"stop commands: {100 * np.mean([r[3] for r in rows]):.1f}% of frames")


def parse_bag_spec(spec):
    parts = spec.split(":")
    idx = int(parts[0])
    start = int(parts[1]) if len(parts) > 1 and parts[1] else 0
    stop = int(parts[2]) if len(parts) > 2 and parts[2] else None
    return idx, start, stop


if __name__ == "__main__":
    import bag_list
    specs = [parse_bag_spec(a) for a in sys.argv[1:]] or [(7, 200, 600), (9, 300, 700)]
    all_rows = []
    for idx, start, stop in specs:
        bag = bag_list.extracted_bags[idx]
        label = f"{os.path.basename(os.path.normpath(bag))} [{start}:{'' if stop is None else stop}]"
        print(f"Processing {label} ...")
        rows = run(bag, start, stop)
        summarise(rows, label)
        all_rows += rows
    summarise(all_rows, "ALL")