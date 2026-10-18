"""Per-frame timing logger for nav_node. No ROS imports."""
import csv
import datetime
import os


class TimingLogger:
    COLUMNS = ["frame", "stamp_s", "stamp_gap_ms", "callback_ms", "convert_ms",
               "data_process_ms", "snake_plan_ms", "publish_ms", "scan_age_ms", "status"]

    def __init__(self, save_dir="live_timing", flush_every=50):
        os.makedirs(save_dir, exist_ok=True)
        stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.path = os.path.join(save_dir, f"live_timing_{stamp}.csv")
        self._f = open(self.path, "w", newline="")
        self._w = csv.writer(self._f)
        self._w.writerow(self.COLUMNS)
        self._last_stamp = None
        self._n = 0
        self.flush_every = flush_every

    def log(self, frame, stamp_s, timing, status):
        """stamp_s: LiDAR header stamp (s). timing: dict of *_ms values."""
        gap = "" if self._last_stamp is None else f"{(stamp_s - self._last_stamp) * 1000:.1f}"
        self._last_stamp = stamp_s
        f = lambda k: f"{timing[k]:.1f}" if k in timing else ""
        self._w.writerow([frame, f"{stamp_s:.3f}", gap, f("callback_ms"), f("convert_ms"),
                          f("data_process_ms"), f("snake_plan_ms"), f("publish_ms"),
                          f("scan_age_ms"), status])
        self._n += 1
        if self._n % self.flush_every == 0:
            self._f.flush()                       # keep data if the node is killed

    def close(self):
        self._f.flush()
        self._f.close()