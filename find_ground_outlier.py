"""Find the frames with the largest ground tilt errors in a saved ground_accuracy_test run.

usage:  python3 find_ground_outlier.py ground_runs/ground_<stamp>.npz            (list worst frames)
        python3 find_ground_outlier.py ground_runs/ground_<stamp>.npz --show     (also view the worst legacy frame)
"""
import os
import sys
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
for _sub in ("nav_source", "datasets"):
    _p = os.path.join(_HERE, _sub)
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

TOP_N = 5


def load(path):
    with np.load(path, allow_pickle=False) as f:
        return {k: f[k] for k in f.files}


def bag_of_each_frame(frames, labels):
    """Frames increase within a bag, so a drop in frame number marks the start of the next bag."""
    starts = np.r_[0, np.where(np.diff(frames) < 0)[0] + 1]
    which = np.zeros(len(frames), dtype=int)
    for b, s in enumerate(starts):
        which[s:] = b
    if len(starts) != len(labels):
        print(f"warning: found {len(starts)} bag segments but {len(labels)} bag labels")
    return which


def report(res, key, title):
    err = res[key]
    order = np.argsort(np.nan_to_num(err, nan=-1))[::-1][:TOP_N]   # NaN (no reference) sorts last
    labels = [str(b) for b in res["bags"]]
    which = bag_of_each_frame(res["frame"], labels)

    print(f"\n{title}: top {TOP_N} errors")
    print(f"{'bag':<45}{'frame':>6}{'err_old':>9}{'err_new':>9}{'tilt_ref':>10}{'tilt_old':>10}{'tilt_new':>10}  mode")
    for k in order:
        print(f"{labels[which[k]]:<45}{res['frame'][k]:>6}{res['err_old'][k]:9.2f}{res['err_new'][k]:9.2f}"
              f"{res['tilt_ref'][k]:10.2f}{res['tilt_old'][k]:10.2f}{res['tilt_new'][k]:10.2f}  {res['mode'][k]}")
    return labels[which[order[0]]], int(res["frame"][order[0]])


def show_frame(bag_label, frame):
    """Re-run the legacy ground fit on one frame and show it side-on (ground green, normal red)."""
    import bag_list
    import local_vis_wrapper
    from lidar_processor import LidarProcessor
    from lidar_scan_storage import ScanStore, scan_to_xyz

    bag_name = bag_label.split(" [")[0]                    # strip the "[start:stop]" part
    bag = next(b for b in bag_list.extracted_bags if os.path.basename(os.path.normpath(b)) == bag_name)

    lp = LidarProcessor(None)
    lp.update_cloud(scan_to_xyz(ScanStore(bag)[frame]))
    lp._LidarProcessor__crop_cloud(False)
    lp._LidarProcessor__reorient_standard(False)
    lp._LidarProcessor__legacy_reorient_RANSAC_only(False)
    print(f"\nre-run legacy fit on {bag_name} frame {frame}: tilt {lp.ground_tilt_deg:.2f} deg")
    local_vis_wrapper.show_ground_reorient_cloud(lp, duration=60, side_on=True)

def show_neighbourhood(res, key="err_old", window=10):
    """Print tilts and tracking mode for the frames around the largest error in `key`."""
    labels = [str(b) for b in res["bags"]]
    which = bag_of_each_frame(res["frame"], labels)
    k = int(np.nanargmax(res[key]))                        # index of the outlier
    print(f"\nframes around the largest {key} ({labels[which[k]]}, frame {res['frame'][k]}):")
    print(f"{'frame':>6}{'tilt_ref':>10}{'tilt_old':>10}{'tilt_new':>10}  mode")
    for j in range(max(0, k - window), min(len(res["frame"]), k + window + 1)):
        if which[j] != which[k]:                           # stay within the same bag
            continue
        mark = "   <-- outlier" if j == k else ""
        print(f"{res['frame'][j]:>6}{res['tilt_ref'][j]:10.2f}{res['tilt_old'][j]:10.2f}"
              f"{res['tilt_new'][j]:10.2f}  {res['mode'][j]}{mark}")

if __name__ == "__main__":
    res = load(sys.argv[1])
    bag_label, frame = report(res, "err_old", "LEGACY method")
    report(res, "err_new", "CURRENT method")
    show_neighbourhood(res, "err_old")
    if "--show" in sys.argv:
        show_frame(bag_label, frame)