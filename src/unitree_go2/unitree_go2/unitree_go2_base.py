# Copyright (c) 2024, RoboVerse community
# SPDX-License-Identifier: BSD-3-Clause
#
# odom -> base_link odometry + TF publisher for a single Go2.
#
# This is the piece SLAM/Nav2 need but nothing else in the bring-up provides:
# a continuous odom -> base_link transform (and a matching nav_msgs/Odometry).
# robot_state_publisher already supplies every base_link -> sensor transform
# from the URDF (base_link -> radar / imu / front_camera), and slam_toolbox
# supplies map -> odom, so once this node closes the odom -> base_link gap the
# full map -> odom -> base_link -> sensor TF chain is connected.
#
# The Go2's onboard sport service publishes its fused odometry as
# unitree_go/msg/SportModeState on the CycloneDDS topic "sportmodestate".
# With the unitree_ros2 bridge sourced that arrives as an ordinary ROS 2 topic,
# so -- like lowstate_joint_publisher and lidar_publisher -- this is a plain
# rclpy subscriber -> publisher. No unitree_sdk2py / ChannelFactoryInitialize
# is used (that opens a second CycloneDDS domain in-process and clashes with
# the ROS RMW).
#
# The TF and Odometry field mapping is taken from go2_ros2_sdk's ROS2Publisher
# (_publish_transform / _publish_odometry_topic).

from dataclasses import dataclass

import rclpy
from rclpy.context import Context
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node

from geometry_msgs.msg import TransformStamped
from nav_msgs.msg import Odometry
from unitree_go.msg import SportModeState

from tf2_ros import TransformBroadcaster


@dataclass
class BaseConfig:
    """Tunables for the odometry/TF publisher. All overridable via ROS params."""

    input_topic: str = 'sportmodestate'
    odom_topic: str = '/odom'
    odom_frame: str = 'odom'
    base_frame: str = 'base_link'
    # Broadcast odom -> base_link on /tf (set False to publish only /odom,
    # e.g. when another node owns that transform).
    publish_tf: bool = True
    # Stamp with the local ROS clock instead of trusting the robot's clock --
    # keeps timestamps monotonic and in sync with the rest of the TF tree.
    use_ros_time: bool = True
    # Lift base_link by the robot's standing height; matches go2_ros2_sdk's
    # +0.07 m offset between the sport-frame origin and the URDF base_link.
    z_offset: float = 0.07


class UnitreeGo2Base(Node):
    """Publishes the Go2 sport odometry as odom -> base_link TF + /odom."""

    def __init__(self, context_in: Context, context_out: Context):
        # The subscription lives on `context_in` (the domain the robot publishes
        # on); the TF/Odometry outputs live on `context_out` (the domain SLAM,
        # Nav2 and robot_state_publisher run on). When the two domains differ
        # these are distinct DDS participants, so we cross domains in-process by
        # reading on one and writing on the other -- same as lidar_publisher.
        super().__init__('unitree_go2_base', context=context_in)

        defaults = BaseConfig()
        self.cfg = BaseConfig(
            input_topic=self.declare_parameter(
                'input_topic', defaults.input_topic).value,
            odom_topic=self.declare_parameter(
                'odom_topic', defaults.odom_topic).value,
            odom_frame=self.declare_parameter(
                'odom_frame', defaults.odom_frame).value,
            base_frame=self.declare_parameter(
                'base_frame', defaults.base_frame).value,
            publish_tf=self.declare_parameter(
                'publish_tf', defaults.publish_tf).value,
            use_ros_time=self.declare_parameter(
                'use_ros_time', defaults.use_ros_time).value,
            z_offset=self.declare_parameter(
                'z_offset', defaults.z_offset).value,
        )

        # The publisher + TF broadcaster are hosted on a separate node bound to
        # `context_out` so they announce on the output domain. Publishing does
        # not require the output context to be spun -- a DDS write is direct --
        # so the executor only needs to spin the input (subscription) node.
        self._pub_node = Node('unitree_go2_base_pub', context=context_out)
        self.odom_pub = self._pub_node.create_publisher(
            Odometry, self.cfg.odom_topic, 10
        )
        self.broadcaster = TransformBroadcaster(self._pub_node)

        self.sub = self.create_subscription(
            SportModeState, self.cfg.input_topic, self._on_state, 10
        )

        in_domain = context_in.get_domain_id()
        out_domain = context_out.get_domain_id()
        self.get_logger().info(
            f'unitree_go2_base up: {self.cfg.input_topic} (domain {in_domain}) '
            f'-> {self.cfg.odom_frame} -> {self.cfg.base_frame} TF + '
            f'{self.cfg.odom_topic} (domain {out_domain}, '
            f'tf={"on" if self.cfg.publish_tf else "off"})'
        )

    def destroy_node(self):
        self._pub_node.destroy_node()
        super().destroy_node()

    def _on_state(self, msg: SportModeState):
        stamp = self.get_clock().now().to_msg()

        pos = msg.position            # float32[3] x, y, z in the odom frame
        # Unitree IMU quaternion is ordered [w, x, y, z]; ROS wants x, y, z, w.
        qw, qx, qy, qz = (float(v) for v in msg.imu_state.quaternion)
        vel = msg.velocity            # float32[3] body-frame linear velocity

        if self.cfg.publish_tf:
            t = TransformStamped()
            t.header.stamp = stamp
            t.header.frame_id = self.cfg.odom_frame
            t.child_frame_id = self.cfg.base_frame
            t.transform.translation.x = float(pos[0])
            t.transform.translation.y = float(pos[1])
            t.transform.translation.z = float(pos[2]) + self.cfg.z_offset
            t.transform.rotation.x = qx
            t.transform.rotation.y = qy
            t.transform.rotation.z = qz
            t.transform.rotation.w = qw
            self.broadcaster.sendTransform(t)

        odom = Odometry()
        odom.header.stamp = stamp
        odom.header.frame_id = self.cfg.odom_frame
        odom.child_frame_id = self.cfg.base_frame
        odom.pose.pose.position.x = float(pos[0])
        odom.pose.pose.position.y = float(pos[1])
        odom.pose.pose.position.z = float(pos[2]) + self.cfg.z_offset
        odom.pose.pose.orientation.x = qx
        odom.pose.pose.orientation.y = qy
        odom.pose.pose.orientation.z = qz
        odom.pose.pose.orientation.w = qw
        # Twist is expressed in child_frame_id (base_link), i.e. body frame --
        # which is exactly how the sport service reports velocity / yaw_speed.
        odom.twist.twist.linear.x = float(vel[0])
        odom.twist.twist.linear.y = float(vel[1])
        odom.twist.twist.linear.z = float(vel[2])
        odom.twist.twist.angular.z = float(msg.yaw_speed)
        self.odom_pub.publish(odom)


def main(args=None):
    # The robot publishes sportmodestate on domain 0; the odom TF / topic are
    # published on domain 1 (where SLAM/Nav2/robot_state_publisher live).
    # Domains have to be fixed before any node exists, so we run two DDS
    # participants -- one per domain -- and bridge between them in-process.
    context_in = Context()
    rclpy.init(args=args, context=context_in, domain_id=0)

    context_out = Context()
    rclpy.init(args=args, context=context_out, domain_id=1)

    node = UnitreeGo2Base(context_in, context_out)
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
