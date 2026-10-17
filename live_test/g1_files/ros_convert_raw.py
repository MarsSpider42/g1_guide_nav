"""Fast CustomMsg decoding straight from the raw CDR bytes (no per-point Python objects)."""
import struct
import numpy as np

_POINT = np.dtype({"names": ["offset_time", "x", "y", "z", "reflectivity", "tag", "line"],
                   "formats": ["<u4", "<f4", "<f4", "<f4", "u1", "u1", "u1"],
                   "offsets": [0, 4, 8, 12, 16, 17, 18],
                   "itemsize": 20})          # 19 bytes of data + 1 byte CDR padding


def custommsg_raw_to_xyz(raw):
    """raw CDR bytes of livox_ros_driver2/CustomMsg -> (N, 3) float64, valid returns (tag == 0)."""
    buf = bytes(raw)
    if buf[1] != 1:
        raise ValueError("big-endian CDR not supported")
    pos = 4                                           # skip the 4-byte encapsulation header
    # alignment in CDR is measured from the start of the data, i.e. from byte 4
    def align(p, n):
        return 4 + ((p - 4 + n - 1) // n) * n
    pos += 8                                          # header.stamp.sec, .nanosec
    (slen,) = struct.unpack_from("<I", buf, pos); pos += 4 + slen      # header.frame_id
    pos = align(pos, 8) + 8                           # timebase (uint64)
    pos += 4 + 1 + 3                                  # point_num, lidar_id, rsvd[3]
    pos = align(pos, 4)
    (n,) = struct.unpack_from("<I", buf, pos); pos += 4
    need = n * _POINT.itemsize
    chunk = buf[pos:pos + need]
    if len(chunk) < need:                             # last element has no trailing padding
        chunk = chunk + b"\0" * (need - len(chunk))
    pts = np.frombuffer(chunk, dtype=_POINT, count=n)
    pts = pts[pts["tag"] == 0]
    return np.stack([pts["x"], pts["y"], pts["z"]], axis=1).astype(np.float64)


def custommsg_raw_stamp(raw):
    """header.stamp of a raw CustomMsg, in seconds (float)."""
    sec, nsec = struct.unpack_from("<iI", bytes(raw[4:12]))
    return sec + nsec * 1e-9
