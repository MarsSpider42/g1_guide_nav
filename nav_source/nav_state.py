import base64
import json
import numpy as np
from snake_planner import Radial_Rail
from path_follower import DriveCommand


def pack_state(frame, stamp_ns, res):
    grid, cmd, tgt, err = res["grid"], res["cmd"], res["target_xy"], res["err"]
    return json.dumps({
        "frame": frame, "stamp_ns": stamp_ns,
        "num_rails": res["num_rails"], "reach_m": res["reach_m"],
        "grid_shape": list(grid.shape),
        "grid": base64.b64encode(np.packbits(grid.astype(np.uint8).ravel()).tobytes()).decode(),
        "rails": [[float(r.coordinate), float(r.lower), float(r.upper),
                   bool(r.valid), bool(r.left_closed), bool(r.right_closed)] for r in res["rails"]],
        "vertices": [float(v) for v in res["vertices"]],
        "target_xy": None if tgt is None else [float(tgt[0]), float(tgt[1])],
        "err": None if err is None else float(err),
        "cmd": [float(cmd.vx), float(cmd.vy), float(cmd.omega), cmd.status],
        "timing": {k: float(v) for k, v in res["timing"].items()},
    })


def unpack_state(text):
    d = json.loads(text)
    rows, cols = d["grid_shape"]
    bits = np.unpackbits(np.frombuffer(base64.b64decode(d["grid"]), dtype=np.uint8))
    d["grid"] = bits[: rows * cols].reshape(rows, cols)
    d["rails"] = [Radial_Rail(c, -lo, hi, v, lc, rc) for c, lo, hi, v, lc, rc in d["rails"]]
    d["vertices"] = np.array(d["vertices"])
    d["cmd"] = DriveCommand(*d["cmd"])
    d["target_xy"] = None if d["target_xy"] is None else tuple(d["target_xy"])
    return d
