# Copyright (c) 2024, RoboVerse community
# SPDX-License-Identifier: BSD-3-Clause
#
# Minimal /joint_states publisher for a single Go2.
#
# The robot's LowState is delivered as an ordinary ROS 2 topic ("lowstate")
# by the unitree_ros2 / CycloneDDS bridge, so this is just a plain rclpy
# subscriber -> publisher. No unitree_sdk2py / ChannelFactoryInitialize is
# used (that creates a second CycloneDDS domain in-process and clashes with
# the ROS RMW). The joint-name + motor-index mapping is taken verbatim from
# go2_ros2_sdk's ROS2Publisher.publish_joint_state.

import rclpy
from rclpy.context import Context
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node

from sensor_msgs.msg import JointState
from unitree_go.msg import LowState


class LowStateJointPublisher(Node):
    """Republishes Go2 LowState motor positions as /joint_states.

    The robot's LowState arrives on the input domain (0); /joint_states is
    published on the output domain (1). When the two domains differ they are
    distinct DDS participants, so -- like lidar_publisher -- we read on one
    context and write on the other, bridging domains in-process.
    """

    def __init__(self, context_in: Context, context_out: Context):
        super().__init__("lowstate_joint_publisher", context=context_in)

        self.joint_names = [
            "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
            "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
            "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
            "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
        ]
        # motor_state index for each joint above (Go2 LegID order:
        # FR=0,1,2  FL=3,4,5  RR=6,7,8  RL=9,10,11; _0=hip _1=thigh _2=calf)
        self.motor_idx = [3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8]

        # The publisher is hosted on a separate node bound to `context_out` so
        # it announces /joint_states on the output domain; the subscription
        # lives on this node (context_in). Only the input node needs spinning --
        # a DDS write is direct and does not require the output context to spin.
        self._pub_node = Node("lowstate_joint_publisher_pub", context=context_out)
        self.pub = self._pub_node.create_publisher(JointState, "/joint_states", 10)
        self.sub = self.create_subscription(
            LowState, "lowstate", self._on_lowstate, 10
        )

        in_domain = context_in.get_domain_id()
        out_domain = context_out.get_domain_id()
        self.get_logger().info(
            f"lowstate_joint_publisher up: lowstate (domain {in_domain}) "
            f"-> /joint_states (domain {out_domain})"
        )

    def destroy_node(self):
        self._pub_node.destroy_node()
        super().destroy_node()

    def _on_lowstate(self, msg: LowState):
        ms = msg.motor_state
        js = JointState()
        js.header.stamp = self.get_clock().now().to_msg()
        js.name = self.joint_names
        js.position = [float(ms[i].q) for i in self.motor_idx]
        js.velocity = [float(ms[i].dq) for i in self.motor_idx]
        js.effort = [float(ms[i].tau_est) for i in self.motor_idx]
        self.pub.publish(js)


def main(args=None):
    # LowState arrives on domain 0; /joint_states is published on domain 1.
    # Domains have to be fixed before any node exists, so we run two DDS
    # participants -- one per domain -- and bridge between them in-process.
    context_in = Context()
    rclpy.init(args=args, context=context_in, domain_id=0)

    context_out = Context()
    rclpy.init(args=args, context=context_out, domain_id=1)

    node = LowStateJointPublisher(context_in, context_out)
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


if __name__ == "__main__":
    main()
