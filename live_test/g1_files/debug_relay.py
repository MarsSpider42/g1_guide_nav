"""Robot side: /g1_nav/debug (ROS 2) -> TCP on 127.0.0.1:5577, for the laptop to reach through an SSH tunnel."""
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from std_msgs.msg import String

from tcp_broadcast import TcpBroadcaster

QOS = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=1)


class Relay(Node):
    def __init__(self, port=5577):
        super().__init__("g1_debug_relay")
        self.server = TcpBroadcaster(port=port)
        self.create_subscription(String, "/g1_nav/debug", self.cb, QOS)
        self.get_logger().info(f"relaying /g1_nav/debug to tcp://127.0.0.1:{port}")

    def cb(self, msg):
        self.server.send(msg.data)


def main():
    rclpy.init()
    try:
        rclpy.spin(Relay())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
