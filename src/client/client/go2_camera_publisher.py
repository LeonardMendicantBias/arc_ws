from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from unitree_sdk2py.go2.video.video_client import VideoClient

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image
from cv_bridge import CvBridge

import cv2
import numpy as np


class Go2CameraPublisher(Node):

    def __init__(self):
        super().__init__("go2_camera")

        self.pub = self.create_publisher(
            Image,
            "/camera/camera/color/image_raw",
            10
        )

        self.bridge = CvBridge()

        self.client = VideoClient()
        self.client.SetTimeout(3.0)
        self.client.Init()

        self.timer = self.create_timer(
            1.0 / 30.0,
            self.publish_image
        )

    def publish_image(self):

        code, data = self.client.GetImageSample()

        if code != 0:
            return

        frame = cv2.imdecode(
            np.frombuffer(bytes(data), dtype=np.uint8),
            cv2.IMREAD_COLOR
        )

        if frame is None:
            return

        frame = cv2.resize(frame, (320, 240))

        msg = self.bridge.cv2_to_imgmsg(
            frame,
            encoding="bgr8"
        )

        self.pub.publish(msg)


def main():
    # The Unitree SDK uses CycloneDDS on domain 0 (mainboard NIC enP8p1s0).
    # Put ROS 2 on domain 1 (Jetson NIC enxc84d44272aa2) so the two
    # participants don't clash in the same process.
    ChannelFactoryInitialize(0, "enP8p1s0")
    rclpy.init(domain_id=1)
    node = Go2CameraPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
