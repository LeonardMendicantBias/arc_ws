# Copyright (c) 2024, RoboVerse community
# SPDX-License-Identifier: BSD-3-Clause
#
# SLAM-ready lidar publisher for a single Go2.
#
# The Go2's onboard `utlidar` process publishes the L1 lidar as a standard
# sensor_msgs/PointCloud2 on the CycloneDDS topic "utlidar/cloud" (frame_id
# "utlidar_lidar"). With the unitree_ros2 bridge sourced, that arrives as an
# ordinary ROS 2 topic, so -- like lowstate_joint_publisher -- this is a plain
# rclpy subscriber -> publisher. No unitree_sdk2py / ChannelFactoryInitialize
# is used (that opens a second CycloneDDS domain in-process and clashes with
# the ROS RMW).
#
# What this node adds on top of the raw cloud, to make it usable by SLAM:
#   * Re-frames the cloud onto the URDF lidar link ("radar"), which is part of
#     the robot_state_publisher TF tree (base_link -> radar). The raw cloud's
#     "utlidar_lidar" frame has no TF and SLAM/Nav2 would drop it.
#   * Re-stamps with the local ROS clock so SLAM gets monotonic, in-sync
#     timestamps (the robot clock embedded in the raw cloud can drift).
#   * Optional range / height cropping and voxel downsampling to cut the cloud
#     down to something a mapper can chew on in real time.
#
# Design mirrors go2_ros2_sdk's infrastructure/sensors layout: a small config
# dataclass describes the behaviour, the node just applies it. Everything is
# exposed as ROS 2 parameters so it can be retuned from a launch file without
# touching code.

from dataclasses import dataclass

import numpy as np

import rclpy
from rclpy.context import Context
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from sensor_msgs.msg import PointCloud2, PointField
from sensor_msgs_py import point_cloud2


@dataclass
class LidarConfig:
    """Tunables for the lidar republisher. All overridable via ROS params."""

    input_topic: str = 'utlidar/cloud'
    output_topic: str = '/robot0/lidar/points'
    # Frame the republished cloud is stamped with. Must exist in the TF tree;
    # 'radar' is the Go2 URDF lidar link (parent: base_link). Empty string keeps
    # the original frame untouched.
    target_frame: str = 'radar'
    # Stamp with the local ROS clock instead of trusting the robot's clock.
    use_ros_time: bool = True
    # Voxel-grid leaf size in metres; <= 0 disables downsampling.
    voxel_size: float = 0.05
    # Euclidean range gate (metres); <= 0 disables that bound.
    min_range: float = 0.1
    max_range: float = 30.0
    # Height gate in the sensor frame (metres); applied only if min_z < max_z.
    min_z: float = -1.0
    max_z: float = 1.0


class LidarPublisher(Node):
    """Republishes the Go2 utlidar cloud as a clean, SLAM-ready PointCloud2."""

    def __init__(self, context_in: Context, context_out: Context):
        # The subscription lives on `context_in` (the domain the lidar arrives
        # on); the publisher lives on `context_out` (the output domain). When
        # the two domains differ these are distinct DDS participants, so we
        # cross domains in-process by reading on one and writing on the other.
        super().__init__('lidar_publisher', context=context_in)

        defaults = LidarConfig()
        self.cfg = LidarConfig(
            input_topic=self.declare_parameter(
                'input_topic', defaults.input_topic).value,
            output_topic=self.declare_parameter(
                'output_topic', defaults.output_topic).value,
            target_frame=self.declare_parameter(
                'target_frame', defaults.target_frame).value,
            use_ros_time=self.declare_parameter(
                'use_ros_time', defaults.use_ros_time).value,
            voxel_size=self.declare_parameter(
                'voxel_size', defaults.voxel_size).value,
            min_range=self.declare_parameter(
                'min_range', defaults.min_range).value,
            max_range=self.declare_parameter(
                'max_range', defaults.max_range).value,
            min_z=self.declare_parameter('min_z', defaults.min_z).value,
            max_z=self.declare_parameter('max_z', defaults.max_z).value,
        )

        # A cloud is only rebuilt (points read + filtered) when some filter is
        # actually enabled; otherwise we just rewrite the header, which is far
        # cheaper and avoids touching the point payload at all.
        self._filtering = (
            self.cfg.voxel_size > 0.0
            or self.cfg.min_range > 0.0
            or self.cfg.max_range > 0.0
            or self.cfg.min_z < self.cfg.max_z
        )

        # Sensor data QoS (best-effort, volatile) on both ends -- matches what
        # the robot publishes and what SLAM/Nav2 expect for point clouds.
        #
        # The publisher is hosted on a separate node bound to `context_out` so
        # it announces on the output domain. Publishing does not require the
        # output context to be spun -- a DDS write is direct -- so the executor
        # only needs to spin the input (subscription) node.
        self._pub_node = Node('lidar_publisher_pub', context=context_out)
        self.pub = self._pub_node.create_publisher(
            PointCloud2, self.cfg.output_topic, qos_profile_sensor_data
        )
        self.sub = self.create_subscription(
            PointCloud2, self.cfg.input_topic, self._on_cloud,
            qos_profile_sensor_data
        )

        in_domain = context_in.get_domain_id()
        out_domain = context_out.get_domain_id()
        self.get_logger().info(
            f'lidar_publisher up: {self.cfg.input_topic} (domain {in_domain}) '
            f'-> {self.cfg.output_topic} (domain {out_domain}, '
            f'frame={self.cfg.target_frame or "<keep>"}, '
            f'filtering={"on" if self._filtering else "off"})'
        )

    def destroy_node(self):
        self._pub_node.destroy_node()
        super().destroy_node()

    def _on_cloud(self, msg: PointCloud2):
        out = self._filter(msg) if self._filtering else msg

        if self.cfg.target_frame:
            out.header.frame_id = self.cfg.target_frame
        if self.cfg.use_ros_time:
            out.header.stamp = self.get_clock().now().to_msg()

        self.pub.publish(out)

    def _filter(self, msg: PointCloud2) -> PointCloud2:
        """Crop + voxel-downsample, preserving intensity when present."""
        has_intensity = any(f.name == 'intensity' for f in msg.fields)
        field_names = ('x', 'y', 'z', 'intensity') if has_intensity \
            else ('x', 'y', 'z')

        # read_points (not read_points_numpy) returns a *structured* array and
        # tolerates clouds whose fields have differing datatypes -- the Go2
        # cloud mixes FLOAT32 xyz with a non-float intensity/ring field, which
        # trips read_points_numpy's single-datatype assertion. We assemble our
        # own contiguous float32 array from the named fields we care about.
        rec = point_cloud2.read_points(
            msg, field_names=field_names, skip_nans=True
        )
        if rec.shape[0] == 0:
            return msg

        pts = np.empty((rec.shape[0], len(field_names)), dtype=np.float32)
        for i, name in enumerate(field_names):
            pts[:, i] = rec[name].astype(np.float32)

        xyz = pts[:, :3]
        keep = np.ones(pts.shape[0], dtype=bool)

        if self.cfg.min_range > 0.0 or self.cfg.max_range > 0.0:
            r = np.linalg.norm(xyz, axis=1)
            if self.cfg.min_range > 0.0:
                keep &= r >= self.cfg.min_range
            if self.cfg.max_range > 0.0:
                keep &= r <= self.cfg.max_range

        if self.cfg.min_z < self.cfg.max_z:
            keep &= (xyz[:, 2] >= self.cfg.min_z) & (xyz[:, 2] <= self.cfg.max_z)

        pts = pts[keep]

        if self.cfg.voxel_size > 0.0 and pts.shape[0] > 0:
            pts = self._voxel_downsample(pts, self.cfg.voxel_size)

        fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
        ]
        if has_intensity:
            fields.append(PointField(
                name='intensity', offset=12,
                datatype=PointField.FLOAT32, count=1))

        cloud = point_cloud2.create_cloud(
            msg.header, fields, pts.astype(np.float32)
        )
        return cloud

    @staticmethod
    def _voxel_downsample(pts: np.ndarray, leaf: float) -> np.ndarray:
        """Keep one representative point per occupied voxel cell."""
        keys = np.floor(pts[:, :3] / leaf).astype(np.int64)
        _, idx = np.unique(keys, axis=0, return_index=True)
        return pts[np.sort(idx)]


def main(args=None):
    # The lidar arrives on domain 0; the cleaned cloud is published on domain 1.
    # Domains have to be fixed before any node exists, so we run two DDS
    # participants -- one per domain -- and bridge between them in-process.
    context_in = Context()
    rclpy.init(args=args, context=context_in, domain_id=0)

    context_out = Context()
    rclpy.init(args=args, context=context_out, domain_id=1)

    node = LidarPublisher(context_in, context_out)
    executor = SingleThreadedExecutor(context=context_in)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if context_in.ok():
            rclpy.shutdown(context=context_in)
        if context_out.ok():
            rclpy.shutdown(context=context_out)


if __name__ == '__main__':
    main()
