"""Listen to /g1_nav/debug for N seconds and report how many frames the nav node really produced."""
import json
import sys
import time
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from std_msgs.msg import String

QOS = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=1)
SECONDS = float(sys.argv[1]) if len(sys.argv) > 1 else 20.0


class Checker(Node):
    def __init__(self):
        super().__init__("check_debug_rate")
        self.rx, self.first, self.last = 0, None, None
        self.create_subscription(String, "/g1_nav/debug", self.cb, QOS)

    def cb(self, msg):
        d = json.loads(msg.data)
        self.rx += 1
        point = (d["frame"], d["stamp_ns"])
        self.first = self.first or point
        self.last = (point, d["cmd"], d["timing"])


def main():
    rclpy.init()
    node = Checker()
    end = time.time() + SECONDS
    while time.time() < end:
        rclpy.spin_once(node, timeout_sec=0.1)
    if node.rx < 2:
        print(f"only {node.rx} message(s) received in {SECONDS:.0f} s: is nav_node.py running, same ROS_DOMAIN_ID?")
        return
    (f0, s0), ((f1, s1), cmd, timing) = node.first, node.last
    secs = (s1 - s0) / 1e9
    print(f"{SECONDS:.0f} s of listening: received {node.rx} messages ({node.rx / SECONDS:.1f} Hz at this subscriber)")
    print(f"node produced frames {f0}..{f1} over {secs:.1f} s of LiDAR time -> {(f1 - f0) / secs:.1f} Hz")
    print(f"messages seen / frames produced: {node.rx} / {f1 - f0 + 1}")
    print("last cmd:", cmd)
    print("last timing:", {k: round(v, 1) for k, v in timing.items()})


if __name__ == "__main__":
    main()
