"""Path follower analysis: mode distribution, fallback guidance points, and mode switching
with vs without smoothing / hysteresis (same rails and snake fed to each variant).

usage:  python3 follower_check.py 7 9:0:2300        (bag_list index[:start[:stop]])
"""
import os
import sys
from dataclasses import replace
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
for _sub in ("nav_source", "datasets"):
    _p = os.path.join(_HERE, _sub)
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

from nav_pipeline import NavPipeline
from path_follower import PathFollower, DriveConfig
from lidar_scan_storage import ScanStore, scan_to_xyz

FRAME_RATE_HZ = 10.0
BASE = DriveConfig()
VARIANTS = {
    "as designed":       BASE,
    "no smoothing":      replace(BASE, smoothing=1.0),
    "no hysteresis":     replace(BASE, turn_off_angle=BASE.turn_on_angle),
    "neither":           replace(BASE, smoothing=1.0, turn_off_angle=BASE.turn_on_angle),
}


def mode_of(cmd):
    if cmd.status != "ok":
        return "stop"
    if cmd.vx > 0 and cmd.omega != 0:
        return "arc"
    if cmd.vx > 0:
        return "drive"
    return "pivot" if cmd.omega != 0 else "stop"


def guidance_rail(rails, target_xy):
    """Index of the rail the guidance point sits on (rail radius closest to the target's distance)."""
    if target_xy is None or not rails:
        return None
    d = np.hypot(*target_xy)
    return int(np.argmin([abs(r.radius_m - d) for r in rails]))


def run(bag_path, start, stop):
    store = ScanStore(bag_path)
    stop = len(store) if stop is None else min(stop, len(store))
    pipe = NavPipeline()                                       # provides grid, rails, snake
    followers = {name: PathFollower(cfg) for name, cfg in VARIANTS.items()}
    modes = {name: [] for name in VARIANTS}
    fallback, with_target = 0, 0

    for i in range(start, stop):
        res = pipe.process(scan_to_xyz(store[i]))
        grid, rails, vertices = res["grid"], res["rails"], res["vertices"]
        for name, f in followers.items():
            if rails:
                cmd = f.compute(rails, vertices, grid)
            else:
                cmd = f._stop("processing failed")
            modes[name].append(mode_of(cmd))
            if name == "as designed" and f.target_xy is not None:
                with_target += 1
                if guidance_rail(rails, f.target_xy) != BASE.guidance_rail:
                    fallback += 1
    return modes, fallback, with_target


def summarise(modes, fallback, with_target, label):
    n = len(modes["as designed"])
    minutes = n / FRAME_RATE_HZ / 60.0
    print(f"\n===== {label} =====  ({n} frames, {minutes:.1f} min)")
    m = np.array(modes["as designed"])
    print("mode share (as designed): " + " | ".join(
        f"{k} {100 * np.mean(m == k):.1f}%" for k in ("drive", "arc", "pivot", "stop")))
    if with_target:
        print(f"fallback guidance point (not on rail {BASE.guidance_rail}): "
              f"{fallback}/{with_target} ({100 * fallback / with_target:.1f}%) of moving frames")
    print("mode switches per minute:")
    for name, seq in modes.items():
        switches = sum(1 for a, b in zip(seq[:-1], seq[1:]) if a != b)
        print(f"  {name:<14} {switches:5d} switches  ({switches / minutes:6.1f} /min)")


def parse_bag_spec(spec):
    parts = spec.split(":")
    idx = int(parts[0])
    start = int(parts[1]) if len(parts) > 1 and parts[1] else 0
    stop = int(parts[2]) if len(parts) > 2 and parts[2] else None
    return idx, start, stop


if __name__ == "__main__":
    import bag_list
    specs = [parse_bag_spec(a) for a in sys.argv[1:]] or [(7, 0, None), (9, 0, 2300)]
    totals = {name: [] for name in VARIANTS}
    fb_total = tgt_total = 0
    for idx, start, stop in specs:
        bag = bag_list.extracted_bags[idx]
        label = f"{os.path.basename(os.path.normpath(bag))} [{start}:{'' if stop is None else stop}]"
        print(f"Processing {label} ...")
        modes, fb, tgt = run(bag, start, stop)
        summarise(modes, fb, tgt, label)
        for name in VARIANTS:
            totals[name] += modes[name]        # note: concatenation adds one artificial boundary switch per bag
        fb_total += fb
        tgt_total += tgt
    summarise(totals, fb_total, tgt_total, "ALL")