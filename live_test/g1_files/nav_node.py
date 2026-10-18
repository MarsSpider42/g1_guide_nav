"""Robot-side node: Livox CustomMsg (raw bytes) -> NavPipeline -> JSON on /g1_nav/debug.
Display only: nothing here sends a command to the robot."""
import time
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from std_msgs.msg import String

from livox_ros_driver2.msg import CustomMsg
from ros_convert_raw import custommsg_raw_to_xyz, custommsg_raw_stamp
from nav_pipeline import NavPipeline
from nav_state import pack_state
from timing_logger import TimingLogger                                     # NEW

LATEST_ONLY = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                         history=HistoryPolicy.KEEP_LAST, depth=1)


class NavNode(Node):
    def __init__(self):
        super().__init__("g1_nav")
        self.declare_parameter("input_topic", "/livox/lidar")
        self.declare_parameter("log_timing", True)
        self.declare_parameter("timing_dir", "/home/unitree/QCR_G1/live_timing")                         # NEW
        topic = self.get_parameter("input_topic").value

        self.pipeline = NavPipeline()
        self.debug_pub = self.create_publisher(String, "/g1_nav/debug", LATEST_ONLY)
        self.create_subscription(CustomMsg, topic, self.on_scan, LATEST_ONLY, raw=True)
        self.frame = 0

        self.timing_log = None
        if self.get_parameter("log_timing").value:
            self.timing_log = TimingLogger(save_dir=self.get_parameter("timing_dir").value)
            self.get_logger().info(f"logging timing to {self.timing_log.path}")

    def on_scan(self, raw):
        t0 = time.perf_counter()
        points = custommsg_raw_to_xyz(raw)
        convert_ms = (time.perf_counter() - t0) * 1000

        res = self.pipeline.process(points)
        res["timing"]["convert_ms"] = convert_ms
        res["timing"]["callback_ms"] = (time.perf_counter() - t0) * 1000  # NEW: conversion + pipeline
        if res["cmd"].status != "ok" and self.frame % 20 == 0:      # don't flood the log
            self.get_logger().warn(res["cmd"].status)

        stamp = custommsg_raw_stamp(raw)
        res["timing"]["scan_age_ms"] = (time.time() - stamp) * 1000   # robot clock vs driver stamp

        t_pub = time.perf_counter()                                        # NEW
        self.debug_pub.publish(String(data=pack_state(self.frame, int(stamp * 1e9), res)))
        res["timing"]["publish_ms"] = (time.perf_counter() - t_pub) * 1000  # NEW: debug display only

        if self.timing_log:                                                # NEW
            self.timing_log.log(self.frame, stamp, res["timing"], res["cmd"].status)
        self.frame += 1


def main():
    rclpy.init()
    node = NavNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    if node.timing_log:                                                    # NEW: flush the CSV
        node.timing_log.close()
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()