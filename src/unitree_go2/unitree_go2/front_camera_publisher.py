import os

import cv2
import numpy as np
import yaml

import rclpy
from rclpy.node import Node

from ament_index_python.packages import get_package_share_directory
from sensor_msgs.msg import CameraInfo, Image
from cv_bridge import CvBridge

from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from unitree_sdk2py.go2.video.video_client import VideoClient


class FrontCameraPublisher(Node):

    def __init__(self):
        super().__init__('front_camera_publisher')
        package_name = 'unitree_go2'
        height = '720'
        # Frame id stamped on both the image and the camera info.
        self.camera_frame_id = 'front_camera'

        self.bridge = CvBridge()

        self.client = VideoClient()
        self.client.SetTimeout(3.0)
        self.client.Init()

        self.timer = self.create_timer(
            1.0 / 30.0,
            self.publish_image
        )

        self.image_pub = self.create_publisher(
            Image,
            "/robot0/camera/color/image_raw",
            10
        )
        self.info_pub = self.create_publisher(
            CameraInfo,
            "/robot0/camera/color/camera_info",
            10
        )

        # Camera info, loaded once from the calibration yaml. Frames are
        # published at the native 1280x720 resolution, which matches the
        # calibration, so the intrinsics are used as-is.
        yaml_file = os.path.join(
            get_package_share_directory(package_name),
            "calibration",
            f"front_camera_{height}.yaml"
        )
        self.get_logger().info(f"Loading camera info from file: {yaml_file}")

        with open(yaml_file, "r") as file_handle:
            camera_data = yaml.safe_load(file_handle)

        self.camera_info = self._build_camera_info(camera_data)

    def _build_camera_info(self, camera_data):
        """Build a CameraInfo from the calibration. Frames are published at the
        calibrated 1280x720 resolution, so the intrinsics need no scaling."""
        camera_info = CameraInfo()
        camera_info.header.frame_id = self.camera_frame_id
        camera_info.width = camera_data["image_width"]
        camera_info.height = camera_data["image_height"]
        camera_info.distortion_model = camera_data["distortion_model"]
        camera_info.d = list(camera_data["distortion_coefficients"]["data"])
        camera_info.k = list(camera_data["camera_matrix"]["data"])
        camera_info.r = list(camera_data["rectification_matrix"]["data"])
        camera_info.p = list(camera_data["projection_matrix"]["data"])
        return camera_info

    def publish_image(self):
        code, data = self.client.GetImageSample()
        if code != 0: return

        frame = cv2.imdecode(
            np.frombuffer(bytes(data), dtype=np.uint8),
            cv2.IMREAD_COLOR
        )
        if frame is None: return

        msg = self.bridge.cv2_to_imgmsg(frame, encoding="bgr8")

        # Stamp the image and camera info with the same time/frame so
        # downstream consumers can associate them.
        stamp = self.get_clock().now().to_msg()
        msg.header.stamp = stamp
        msg.header.frame_id = self.camera_frame_id
        self.camera_info.header.stamp = stamp

        self.image_pub.publish(msg)
        self.info_pub.publish(self.camera_info)


def main(args=None):
    # The Unitree SDK uses CycloneDDS on domain 0 (mainboard NIC enP8p1s0).
    # This must be initialized before any VideoClient is created, otherwise
    # the channel factory participant is None. Put ROS 2 on domain 1 so the
    # two DDS participants don't clash in the same process.
    ChannelFactoryInitialize(0, "enP8p1s0")
    rclpy.init(domain_id=1)

    front_camera_publisher = FrontCameraPublisher()

    try:
        rclpy.spin(front_camera_publisher)
    except KeyboardInterrupt:
        pass
    finally:
        front_camera_publisher.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
