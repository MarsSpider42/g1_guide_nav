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

LATEST_ONLY = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                         history=HistoryPolicy.KEEP_LAST, depth=1)


class NavNode(Node):
    def __init__(self):
        super().__init__("g1_nav")
        self.declare_parameter("input_topic", "/livox/lidar")
        topic = self.get_parameter("input_topic").value

        self.pipeline = NavPipeline()
        self.debug_pub = self.create_publisher(String, "/g1_nav/debug", LATEST_ONLY)
        self.create_subscription(CustomMsg, topic, self.on_scan, LATEST_ONLY, raw=True)
        self.frame = 0
        self.get_logger().info(f"g1_nav listening on {topic}")

    def on_scan(self, raw):
        t0 = time.perf_counter()
        points = custommsg_raw_to_xyz(raw)
        convert_ms = (time.perf_counter() - t0) * 1000

        res = self.pipeline.process(points)
        res["timing"]["convert_ms"] = convert_ms
        if res["cmd"].status != "ok" and self.frame % 20 == 0:      # don't flood the log
            self.get_logger().warn(res["cmd"].status)

        stamp = custommsg_raw_stamp(raw)
        res["timing"]["scan_age_ms"] = (time.time() - stamp) * 1000   # robot clock vs driver stamp
        self.debug_pub.publish(String(data=pack_state(self.frame, int(stamp * 1e9), res)))
        self.frame += 1


def main():
    rclpy.init()
    node = NavNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
