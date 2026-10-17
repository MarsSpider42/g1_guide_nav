from pathlib import Path
import json
import numpy as np

# rosbags is only needed to read or extract from a bag. Reading already-extracted scans
# (ScanStore, scan_to_xyz) needs only numpy, so a missing rosbags (e.g. on the robot) is not fatal.
try:
    from rosbags.highlevel import AnyReader
    from rosbags.typesys import Stores, get_typestore, get_types_from_msg
    HAVE_ROSBAGS = True
except ImportError:
    HAVE_ROSBAGS = False


def _require_rosbags():
    if not HAVE_ROSBAGS:
        raise RuntimeError(
            "The 'rosbags' package is not installed (or is too old to provide AnyReader / "
            "get_typestore), so bags cannot be read or extracted on this machine. Reading "
            "already-extracted scans (ScanStore, scan_to_xyz) does not need it. "
            "To read bags, install it with: python3 -m pip install rosbags"
        )


LIDAR_TOPIC_CANDIDATES = [
    ("/livox/lidar", "custom"),
    ("/utlidar/cloud_livox_mid360", "pointcloud2"),
]

POINT_DTYPE = np.dtype([
    ("x", "f4"), ("y", "f4"), ("z", "f4"),
    ("reflectivity", "u1"), ("tag", "u1"), ("line", "u1"),
    ("offset_time", "u4"),
])

INDEX_DTYPE = np.dtype([
    ("bag_timestamp", "i8"),
    ("point_num", "i4"),
])

# --- Register the Livox custom message types (not in rosbags' built-in store) ---
CUSTOM_POINT_MSG = """
uint32 offset_time
float32 x
float32 y
float32 z
uint8 reflectivity
uint8 tag
uint8 line
"""

CUSTOM_MSG_MSG = """
std_msgs/Header header
uint64 timebase
uint32 point_num
uint8 lidar_id
uint8[3] rsvd
livox_ros_driver2/CustomPoint[] points
"""

def _build_typestore():
    _require_rosbags()
    typestore = get_typestore(Stores.ROS2_FOXY)
    types = {}
    types.update(get_types_from_msg(CUSTOM_POINT_MSG, "livox_ros_driver2/msg/CustomPoint"))
    types.update(get_types_from_msg(CUSTOM_MSG_MSG, "livox_ros_driver2/msg/CustomMsg"))
    typestore.register(types)
    return typestore


# Built on first use instead of at import time, so importing this module never needs rosbags.
_TYPESTORE = None


def _get_typestore():
    global _TYPESTORE
    if _TYPESTORE is None:
        _TYPESTORE = _build_typestore()
    return _TYPESTORE


def __getattr__(name):
    # Keeps `from lidar_scan_storage import TYPESTORE` working in other scripts (Python 3.7+).
    if name == "TYPESTORE":
        return _get_typestore()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def _detect_topic(reader):
    available = {c.topic: c.msgtype for c in reader.connections}
    for topic, fmt in LIDAR_TOPIC_CANDIDATES:
        if topic in available:
            return topic, fmt
    raise RuntimeError(f"No known LiDAR topic found. Available: {list(available)}")


def _custom_msg_to_array(msg):
    pts = np.empty(msg.point_num, dtype=POINT_DTYPE)
    for j, p in enumerate(msg.points):
        pts[j] = (p.x, p.y, p.z, p.reflectivity, p.tag, p.line, p.offset_time)
    return pts


def _pointcloud2_to_array(msg):
    # Standard PointCloud2 field layout is already known to the typestore;
    # decode the raw byte buffer using its own field offsets/step.
    fields = {f.name: f for f in msg.fields}
    n = msg.width * msg.height
    pts = np.zeros(n, dtype=POINT_DTYPE)
    data = np.frombuffer(msg.data, dtype=np.uint8).reshape(n, msg.point_step)
    for j in range(n):
        row = data[j]
        x = np.frombuffer(row[fields["x"].offset:fields["x"].offset + 4], dtype="f4")[0]
        y = np.frombuffer(row[fields["y"].offset:fields["y"].offset + 4], dtype="f4")[0]
        z = np.frombuffer(row[fields["z"].offset:fields["z"].offset + 4], dtype="f4")[0]
        pts[j] = (x, y, z, 0, 0, 0, 0)
    return pts


def summarize_bag(bag_path):
    _require_rosbags()
    with AnyReader([Path(bag_path)], default_typestore=_get_typestore()) as reader:
        topic, fmt = _detect_topic(reader)
        connections = [c for c in reader.connections if c.topic == topic]
        n_scans, first_t, last_t = 0, None, None
        for connection, t, rawdata in reader.messages(connections=connections):
            n_scans += 1
            first_t = t if first_t is None else first_t
            last_t = t

    duration_s = (last_t - first_t) / 1e9 if n_scans > 1 else 0.0
    return {
        "topic": topic,
        "format": fmt,
        "num_scans": n_scans,
        "duration_s": duration_s,
        "avg_rate_hz": (n_scans - 1) / duration_s if duration_s > 0 else 0.0,
    }


def extract_scans(bag_path, out_dir):
    _require_rosbags()
    out_dir = Path(out_dir)
    scans_dir = out_dir / "scans"
    scans_dir.mkdir(parents=True, exist_ok=True)

    index_rows = []
    i = 0
    with AnyReader([Path(bag_path)], default_typestore=_get_typestore()) as reader:
        topic, fmt = _detect_topic(reader)
        connections = [c for c in reader.connections if c.topic == topic]
        convert = _custom_msg_to_array if fmt == "custom" else _pointcloud2_to_array

        for connection, t, rawdata in reader.messages(connections=connections):
            msg = reader.deserialize(rawdata, connection.msgtype)
            pts = convert(msg)
            np.save(scans_dir / f"{i:06d}.npy", pts)
            index_rows.append((t, len(pts)))
            i += 1

    index = np.array(index_rows, dtype=INDEX_DTYPE)
    np.save(out_dir / "index.npy", index)
    with open(out_dir / "meta.json", "w") as f:
        json.dump({"source_bag": str(bag_path), "topic": topic, "format": fmt, "num_scans": i}, f, indent=2)

    return i, topic, fmt

def scan_to_xyz(scan, valid_only=False):
    """Convert a structured scan array to a plain (N, 3) float array of x, y, z."""
    if valid_only:
        scan = scan[scan["tag"] == 0]
    return np.stack([scan["x"], scan["y"], scan["z"]], axis=-1).astype(np.float64)


class ScanStore:
    def __init__(self, out_dir):
        out_dir = Path(out_dir).expanduser()
        self.scans_dir = out_dir / "scans"
        self.index = np.load(out_dir / "index.npy")
        meta_path = out_dir / "meta.json"
        self.meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        return np.load(self.scans_dir / f"{i:06d}.npy", mmap_mode="r")

    def timestamp(self, i):
        return int(self.index[i]["bag_timestamp"])

    def point_num(self, i):
        return int(self.index[i]["point_num"])