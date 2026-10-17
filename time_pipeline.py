import contextlib
import datetime
import io
import os
import sys
import time
import numpy as np

# Repo layout: nav_source/ (pipeline), datasets/ (ScanStore). Make both importable from wherever this file is run.
_HERE = os.path.dirname(os.path.abspath(__file__))
for _sub in ("nav_source", "datasets"):
    _p = os.path.join(_HERE, _sub)
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

DEFAULT_SAVE_DIR = os.path.join(_HERE, "timing_runs")     # every run is saved here (see time_pipeline)

# LidarProcessor.process(debug) calls these in this order. They are name-mangled
# (double underscore), so we reach them as _LidarProcessor__name, and each takes (debug).
# If you change process(), update this list too (the cross-check at the end will flag drift).
LP_STAGES = [
    ("crop",             "_LidarProcessor__crop_cloud"),
    ("std reorient",     "_LidarProcessor__reorient_standard"),
    ("ground plane",     "_LidarProcessor__reorient_wrt_ground_plane"),
    ("sections",         "_LidarProcessor__identify_sections"),
    ("height/occ grids", "_LidarProcessor__height_occ_grids"),
]


def timed(fn, *args):
    t0 = time.perf_counter()
    out = fn(*args)
    return out, (time.perf_counter() - t0) * 1000.0     # ms


def plot_timing(run, show=True):
    """Rebuild the stacked per-frame timing graph from a saved run.

    run: the path of a timing_*.npz file written by time_pipeline(), or the dict it built.
    Only numpy and matplotlib are needed, so this works on any machine:
        python3 time_pipeline.py --plot timing_runs/timing_20261018_010400_<bag>.npz
    """
    import matplotlib.pyplot as plt
    if not isinstance(run, dict):
        with np.load(run, allow_pickle=False) as f:
            run = {k: f[k] for k in f.files}
    names = [str(n) for n in run["names"]]
    live = [k for k, n in enumerate(names) if n != "read"]          # bag I/O is not part of a live system
    frames = run["frame_ids"]
    budget_ms = float(run["budget_ms"])

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.stackplot(frames, run["data"][:, live].T, labels=[names[k] for k in live])
    ax.axhline(budget_ms, color="k", linestyle="--", label=f"budget {budget_ms:.0f} ms")
    ax.set_xlabel("frame (after warm-up)")
    ax.set_ylabel("ms")
    ax.set_title(f"Pipeline time per frame ({run['rail_type']} rails) - "
                 f"{os.path.basename(os.path.normpath(str(run['bag_path'])))} - {run['timestamp']}")
    ax.legend(loc="upper right", ncol=2, fontsize=8)
    if show:
        plt.show()
    return fig


def time_pipeline(bag_path, config=None, drive_config=None, rail_type="radial",
                  start_frame=0, num_frames=200, warmup=3, budget_ms=100.0,
                  csv_path=None, plot=False, quiet=True, save_dir=DEFAULT_SAVE_DIR):
    """Time every stage per frame.

    Frames are processed IN ORDER on one LidarProcessor, because the ground stage tracks the previous frame's
    plane: a frame is cheap when the plane still fits and expensive when RANSAC has to search again.
    quiet=True hides the "[ground] ODD" log lines during timing (printing costs time) and just counts them.
    save_dir: every run writes timing_<timestamp>_<bag>.npz here (None = do not save); plot_timing() rebuilds the
    graph from that file later. Returns the path of the saved file.
    """
    from lidar_processor import LidarProcessor
    from lidar_scan_storage import ScanStore, scan_to_xyz
    from snake_planner import (SnakeConfig, SnakePlanner, RadialRailExtractor,
                               StraightRailExtractor)
    from path_follower import PathFollower

    config = config or SnakeConfig(w_external=3, w_internal=90, num_rails=18,
                                   min_dist_m=1, max_dist_m=3.5)
    planner = SnakePlanner(config)
    follower = PathFollower(drive_config)
    extractor = (RadialRailExtractor(config) if rail_type == "radial"
                 else StraightRailExtractor(config))

    store = ScanStore(bag_path)
    LP = LidarProcessor(None)
    stage_fns = [(name, getattr(LP, attr)) for name, attr in LP_STAGES]

    names = (["read", "convert", "update cloud"] + [n for n, _ in LP_STAGES]
             + ["extract rails", "snake plan", "follower"])

    def hush(buf):
        return contextlib.redirect_stdout(buf) if quiet else contextlib.nullcontext()

    # =====================================================
    # TIME EACH FRAME, STAGE BY STAGE
    # =====================================================
    rows, modes, frame_ids, staged_by_frame = [], [], [], {}
    failed, first_err = 0, None
    log = io.StringIO()                         # captures LidarProcessor's own prints while quiet
    last = min(len(store), start_frame + num_frames)

    for i in range(start_frame, last):
        try:
            t = {}
            frame, t["read"] = timed(store.__getitem__, i)
            points, t["convert"] = timed(scan_to_xyz, frame)
            with hush(log):
                _, t["update cloud"] = timed(LP.update_cloud, points)
                for name, fn in stage_fns:
                    _, t[name] = timed(fn, False)

            grid = np.array(LP.occupancy_grid)
            rails, t["extract rails"] = timed(extractor.extract_rails, grid)
            _, t["snake plan"] = timed(planner.plan, rails)
            _, t["follower"] = timed(follower.compute, rails, planner.vertices, grid)

            rows.append([t[n] for n in names])
            modes.append(LP.ground_timing["mode"])          # "track" or "search"
            frame_ids.append(i)
            staged_by_frame[i] = sum(t[n] for n, _ in LP_STAGES)
        except Exception as e:          # e.g. no ground plane found, empty obstacle cloud
            failed += 1
            first_err = first_err or repr(e)

    data = np.array(rows)[warmup:]
    modes = np.array(modes[warmup:])
    frame_ids = np.array(frame_ids[warmup:])
    if len(data) == 0:
        print(f"No frames timed ({failed} failed). First error: {first_err}")
        return

    # "read" is bag I/O, which does not exist in a live system, so it is excluded from the total
    live = [k for k, n in enumerate(names) if n != "read"]
    total = data[:, live].sum(axis=1)

    # =====================================================
    # REPORT
    # =====================================================
    print(f"\n{len(data)} frames timed ({warmup} warm-up dropped, {failed} failed), "
          f"{rail_type} rails, frames {start_frame}-{last - 1}")
    if first_err:
        print(f"first failure: {first_err}")
    print(f"\n{'stage':<18}{'mean':>9}{'median':>9}{'p95':>9}{'max':>9}{'% of total':>12}   (ms)")
    for k, n in enumerate(names):
        col = data[:, k]
        pct = "-" if n == "read" else f"{100 * col.mean() / total.mean():.1f}"
        note = "   offline only" if n == "read" else ""
        print(f"{n:<18}{col.mean():9.1f}{np.median(col):9.1f}"
              f"{np.percentile(col, 95):9.1f}{col.max():9.1f}{pct:>12}{note}")
    print(f"{'PIPELINE TOTAL':<18}{total.mean():9.1f}{np.median(total):9.1f}"
          f"{np.percentile(total, 95):9.1f}{total.max():9.1f}")

    over = 100 * np.mean(total > budget_ms)
    print(f"\nbudget {budget_ms:.0f} ms: {over:.1f}% of frames over | "
          f"mean rate {1000 / total.mean():.1f} Hz | worst case {1000 / total.max():.1f} Hz")

    # ground tracking: how often RANSAC had to search, and what each kind of frame costs
    g = names.index("ground plane")
    searched = modes == "search"
    print(f"\nground plane: {searched.sum()} of {len(modes)} frames ran a full RANSAC search "
          f"({100 * searched.mean():.1f}%), the rest reused the previous plane")
    for label, sel in (("search frames", searched), ("tracked frames", ~searched)):
        if sel.any():
            col = data[sel, g]
            print(f"  {label:<15} mean {col.mean():6.1f}  p95 {np.percentile(col, 95):6.1f}  max {col.max():6.1f} ms")
    odd = log.getvalue().count("[ground] ODD")
    if quiet:
        print(f"  tilt jumps > 2 deg logged by the ground stage: {odd} (hidden while timing; use quiet=False to see them)")

    if csv_path:
        np.savetxt(csv_path, data, delimiter=",", header=",".join(names), comments="")
        print(f"per-frame timings saved to {csv_path}")

    # everything plot_timing() needs, plus the run's settings, saved under a timestamped name
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run = dict(data=data, names=np.array(names), modes=modes, frame_ids=frame_ids,
               budget_ms=budget_ms, rail_type=rail_type, bag_path=str(bag_path),
               start_frame=start_frame, warmup=warmup, failed=failed, odd_logged=odd, timestamp=stamp)
    saved = None
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        saved = os.path.join(save_dir, f"timing_{stamp}_{os.path.basename(os.path.normpath(str(bag_path)))}.npz")
        np.savez(saved, **run)
        print(f"run saved to {saved}  (re-plot: python3 time_pipeline.py --plot {saved})")

    # =====================================================
    # CROSS-CHECK: staged times vs the real process() call
    # =====================================================
    # A fresh instance, same frames in the same order, so ground tracking follows the same path as the timed run.
    LP2 = LidarProcessor(None)
    chk, ref = [], []
    for i in range(start_frame, min(start_frame + 10, len(store))):
        if i not in staged_by_frame:
            continue
        LP2.update_cloud(scan_to_xyz(store[i]))
        t0 = time.perf_counter()
        try:
            with hush(io.StringIO()):
                LP2.process(False)
        except Exception:
            continue
        chk.append((time.perf_counter() - t0) * 1000.0)
        ref.append(staged_by_frame[i])
    if chk:
        print(f"LidarProcessor.process(): {np.mean(chk):.1f} ms vs staged sum {np.mean(ref):.1f} ms "
              f"over the same {len(chk)} frames (should be close; a big gap means LP_STAGES has drifted from process())")

    # =====================================================
    # PLOT
    # =====================================================
    if plot:
        plot_timing(run)
    return saved


if __name__ == "__main__":
    # usage: MPLBACKEND=Agg python3 time_pipeline.py [<extracted scan folder> [start_frame [num_frames]]]
    #        python3 time_pipeline.py --plot <saved timing_*.npz>      (re-draw a saved run's graph)
    # with no arguments it times extracted_bags[10] from bag_list.py
    plot_timing('/home/dream/g1_guide_nav/timing_runs/timing_20261018_012456_lidar_test_bag_261014_2_06pm.npz')
    if len(sys.argv) > 2 and sys.argv[1] == "--plot":
        plot_timing(sys.argv[2])
    elif len(sys.argv) > 1:
        bag = sys.argv[1]
        start = int(sys.argv[2]) if len(sys.argv) > 2 else 0
        num = int(sys.argv[3]) if len(sys.argv) > 3 else 200
        time_pipeline(bag, start_frame=start, num_frames=num, rail_type="radial")
    else:
        import bag_list
        time_pipeline(bag_list.extracted_bags[10], rail_type="radial", plot=True)
        