"""Ground plane accuracy + timing test: current (tracking) method vs legacy (RANSAC every frame).

Both methods are compared against a high-iteration reference fit:
    full multi-plane RANSAC search on the full cloud (no downsampling, many iterations)
    + least-squares refinement of the winning plane.
Error = angle between a method's ground normal and the reference normal (deg).

Frames are processed IN ORDER, because the current method tracks the previous frame's plane.
The reference is expensive, so by default it is only computed every REF_EVERY frames;
accuracy statistics use those frames, timing statistics use every frame.

usage:  MPLBACKEND=Agg python3 ground_accuracy_test.py                  (default bags, all frames)
        python3 ground_accuracy_test.py 7:120:400 9:0:250               (bag_list index:start:stop)
        python3 ground_accuracy_test.py 7 9:50                          (start/stop optional; stop is exclusive)
        python3 ground_accuracy_test.py --plot ground_runs/<file>.npz
"""
import contextlib
import datetime
import io
import os
import sys
import time
import numpy as np
import open3d as o3d

_HERE = os.path.dirname(os.path.abspath(__file__))
for _sub in ("nav_source", "datasets"):
    _p = os.path.join(_HERE, _sub)
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

from lidar_processor import LidarProcessor
from lidar_scan_storage import ScanStore, scan_to_xyz

DEFAULT_SAVE_DIR = os.path.join(_HERE, "ground_runs")
REF_ITERATIONS = 2000      # RANSAC iterations per candidate plane for the reference fit
REF_EVERY = 5              # compute the reference on every Nth frame (it is slow)
WARMUP = 3                 # frames dropped from timing statistics
UP = np.array([0.0, 0.0, 1.0])


# =====================================================
# HELPERS
# =====================================================
def angle_between(n1, n2):
    """Angle (deg) between two plane normals, ignoring sign."""
    if n1 is None or n2 is None:
        return np.nan
    c = abs(float(np.dot(n1, n2)) / (np.linalg.norm(n1) * np.linalg.norm(n2)))
    return float(np.degrees(np.arccos(np.clip(c, 0.0, 1.0))))


def tilt_of(n):
    """Tilt (deg) of a normal from the sensor's vertical axis."""
    return np.nan if n is None else float(np.degrees(np.arccos(np.clip(abs(n[2]) / np.linalg.norm(n), 0, 1))))


def refine_plane(pts, n, d, threshold, passes=3):
    """Least-squares refit of a plane to the points within `threshold` of it (smallest-eigenvalue direction)."""
    n = np.asarray(n, float)
    s = np.linalg.norm(n)
    n, d = n / s, d / s
    for _ in range(passes):
        sel = pts[np.abs(pts @ n + d) < threshold]
        if len(sel) < 50:
            break
        c = sel.mean(axis=0)
        n = np.linalg.eigh((sel - c).T @ (sel - c))[1][:, 0]
        d = -float(n @ c)
    return n if n @ UP >= 0 else -n


def reference_normal(pcd, iterations=REF_ITERATIONS):
    """High-effort ground fit: the same multi-plane search and scoring as the pipeline,
    but on the full cloud with many iterations, then a least-squares refinement."""
    L = LidarProcessor
    fit = o3d.geometry.PointCloud(pcd.points)
    best_model, best_score = None, -np.inf
    for _ in range(L.MAX_PLANES_TO_FIT):
        if len(fit.points) < 500:
            break
        model, inl = fit.segment_plane(distance_threshold=L.GROUND_PLANE_THRESHOLD,
                                       ransac_n=L.RANSAC_N, num_iterations=iterations)
        n = np.asarray(model[:3], float)
        n /= np.linalg.norm(n)
        if n @ UP < 0:
            n = -n
        align = float(n @ UP)
        if align >= L.MIN_ALIGNMENT:                     # skip probable walls, as the pipeline does
            centroid = np.asarray(fit.select_by_index(inl).points).mean(axis=0)
            score = (L.FLAT_PLANES_REWARD_COEF * align + L.LARGE_PLANES_REWARD_COEF * len(inl)
                     + L.LOW_CENTROID_REWARD_COEF * centroid[2])
            if score > best_score:
                best_score, best_model = score, model
        fit = fit.select_by_index(inl, invert=True)
    if best_model is None:
        return None
    return refine_plane(np.asarray(pcd.points), best_model[:3], best_model[3], L.GROUND_PLANE_THRESHOLD)


def prepare(lp, points):
    """Run the stages before the ground fit (crop + fixed mount rotation)."""
    lp.update_cloud(points)
    lp._LidarProcessor__crop_cloud(False)
    lp._LidarProcessor__reorient_standard(False)


def timed_ground(lp, method_attr):
    """Run one ground method; returns (normal or None, ms)."""
    t0 = time.perf_counter()
    try:
        getattr(lp, method_attr)(False)
        n = np.array(lp.best_normal, float)
    except RuntimeError:                                  # "No valid ground plane found"
        n = None
    return n, (time.perf_counter() - t0) * 1000.0


# =====================================================
# TEST
# =====================================================
def run_ground_test(bag_path, start_frame=0, num_frames=None, ref_every=REF_EVERY, quiet=True):
    """Process one bag in order. Returns a dict of per-frame results."""
    try:
        o3d.utility.random.seed(0)                       # repeatable RANSAC (Open3D >= 0.16)
    except AttributeError:
        pass

    store = ScanStore(bag_path)
    last = len(store) if num_frames is None else min(len(store), start_frame + num_frames)
    lp_new, lp_old = LidarProcessor(None), LidarProcessor(None)   # separate instances: no shared state

    cols = {k: [] for k in ("frame", "t_new", "t_old", "mode", "tilt_new", "tilt_old",
                            "tilt_ref", "err_new", "err_old")}
    log = io.StringIO()
    hush = (lambda: contextlib.redirect_stdout(log)) if quiet else contextlib.nullcontext

    for k, i in enumerate(range(start_frame, last)):
        points = scan_to_xyz(store[i])
        prepare(lp_new, points)
        prepare(lp_old, points)

        with hush():
            n_new, t_new = timed_ground(lp_new, "_LidarProcessor__reorient_wrt_ground_plane")
            n_old, t_old = timed_ground(lp_old, "_LidarProcessor__legacy_reorient_RANSAC_only")

        n_ref = reference_normal(lp_old.std_reoriented_cloud) if k % ref_every == 0 else None

        cols["frame"].append(i)
        cols["t_new"].append(t_new)
        cols["t_old"].append(t_old)
        cols["mode"].append(lp_new.ground_mode if n_new is not None else "failed")
        cols["tilt_new"].append(tilt_of(n_new))
        cols["tilt_old"].append(tilt_of(n_old))
        cols["tilt_ref"].append(tilt_of(n_ref))
        cols["err_new"].append(angle_between(n_new, n_ref))
        cols["err_old"].append(angle_between(n_old, n_ref))

        if (k + 1) % 25 == 0:
            print(f"  frame {i} ({k + 1}/{last - start_frame})")

    res = {key: np.array(v) for key, v in cols.items()}
    res["bag_path"] = str(bag_path)
    return res


def summarise(res, label=""):
    """Print accuracy and timing statistics for one result dict (or a merged one)."""
    def stats(x):
        x = x[~np.isnan(x)]
        return (f"median {np.median(x):6.2f}  p95 {np.percentile(x, 95):6.2f}  max {x.max():6.2f}"
                if len(x) else "no data")

    print(f"\n===== {label} =====")
    n_ref = np.sum(~np.isnan(res["tilt_ref"]))
    print(f"frames: {len(res['frame'])} | reference frames: {n_ref}")

    print("\nTILT ERROR vs reference (deg)")
    print(f"  current (tracking) : {stats(res['err_new'])}")
    print(f"  legacy (RANSAC)    : {stats(res['err_old'])}")

    t_new, t_old, mode = res["t_new"][WARMUP:], res["t_old"][WARMUP:], res["mode"][WARMUP:]
    print(f"\nGROUND STAGE TIME (ms, first {WARMUP} frames dropped)")
    print(f"  current (tracking) : mean {t_new.mean():6.1f}  {stats(t_new)}")
    print(f"  legacy (RANSAC)    : mean {t_old.mean():6.1f}  {stats(t_old)}")
    print(f"  speed-up (mean)    : {t_old.mean() / t_new.mean():.1f}x")

    tracked = mode == "track"
    print(f"\ncurrent method: {100 * tracked.mean():.1f}% of frames tracked, "
          f"{100 * (mode == 'search').mean():.1f}% searched, {np.sum(mode == 'failed')} failed")
    for name, sel in (("tracked frames", tracked), ("search frames", mode == "search")):
        if sel.any():
            print(f"  {name:<15} mean {t_new[sel].mean():6.1f} ms")
    print(f"legacy failures: {np.sum(np.isnan(res['tilt_old']))}")


def merge(results):
    keys = [k for k in results[0] if isinstance(results[0][k], np.ndarray)]
    return {k: np.concatenate([r[k] for r in results]) for k in keys}


# =====================================================
# PLOT
# =====================================================
def plot_ground_test(res, save_path=None, show=True):
    """Two panels: (a) tilt error vs reference, (b) ground stage time. res = dict or saved .npz path."""
    import matplotlib.pyplot as plt
    if not isinstance(res, dict):
        with np.load(res, allow_pickle=False) as f:
            res = {k: f[k] for k in f.files}

    err_new = res["err_new"][~np.isnan(res["err_new"])]
    err_old = res["err_old"][~np.isnan(res["err_old"])]
    t_new, t_old = res["t_new"][WARMUP:], res["t_old"][WARMUP:]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))

    ax1.boxplot([err_old, err_new], labels=["Legacy\n(RANSAC every frame)", "Current\n(tracking)"],
                showfliers=True)
    ax1.set_ylabel("Tilt error vs reference (deg)")
    ax1.set_title("(a) Ground plane accuracy")
    ax1.grid(True, axis="y", alpha=0.3)

    ax2.boxplot([t_old, t_new], labels=["Legacy\n(RANSAC every frame)", "Current\n(tracking)"],
                showfliers=True)
    ax2.set_ylabel("Ground stage time per frame (ms)")
    ax2.set_title("(b) Ground stage processing time")
    ax2.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    return fig


# =====================================================
# MAIN
# =====================================================
def parse_bag_spec(spec):
    """'7' -> (7, 0, None); '7:120' -> (7, 120, None); '7:120:400' -> (7, 120, 400). Stop is exclusive."""
    parts = spec.split(":")
    if not 1 <= len(parts) <= 3 or not parts[0]:
        raise ValueError(f"bad bag spec '{spec}': use index[:start[:stop]], e.g. 7:120:400")
    idx = int(parts[0])
    start = int(parts[1]) if len(parts) > 1 and parts[1] else 0
    stop = int(parts[2]) if len(parts) > 2 and parts[2] else None
    if stop is not None and stop <= start:
        raise ValueError(f"bad bag spec '{spec}': stop ({stop}) must be greater than start ({start})")
    return idx, start, stop


def main(bag_specs=((7, 0, None), (9, 0, None)), save_dir=DEFAULT_SAVE_DIR):
    """bag_specs: list of (bag_list index, start_frame, stop_frame or None)."""
    import bag_list
    results, labels = [], []
    for idx, start, stop in bag_specs:
        bag = bag_list.extracted_bags[idx]
        num = None if stop is None else stop - start
        label = (f"{os.path.basename(os.path.normpath(bag))} "
                 f"[{start}:{'' if stop is None else stop}]")
        print(f"\nProcessing {label} ...")
        res = run_ground_test(bag, start_frame=start, num_frames=num)
        summarise(res, label)
        results.append(res)
        labels.append(label)

    allres = merge(results)
    summarise(allres, "ALL BAGS")

    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(save_dir, exist_ok=True)
    out = os.path.join(save_dir, f"ground_{stamp}.npz")
    np.savez(out, **allres, ref_iterations=REF_ITERATIONS, ref_every=REF_EVERY,
             bags=np.array(labels))                       # bag names with the frame ranges used
    print(f"\nsaved {out}  (re-plot: python3 ground_accuracy_test.py --plot {out})")

    plot_ground_test(allres, save_path=out.replace(".npz", ".png"))


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] == "--plot":
        plot_ground_test(sys.argv[2])
    elif len(sys.argv) > 1:
        main(bag_specs=[parse_bag_spec(a) for a in sys.argv[1:]])
    else:
        main()