import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter

import numpy as np

from std_msgs.msg import Header
from rcl_interfaces.msg import SetParametersResult

import torch


class Critic(Node):

	def __init__(self):
		super().__init__('critic')

		self.declare_parameter('metric', '')

		self.add_on_set_parameters_callback(self._on_parameters_changed)

	def _on_parameters_changed(self, params):
		for param in params:
			if param.name == 'metric':
				self.get_logger().info(f"metric updated: '{param.value}'")
		return SetParametersResult(successful=True)


def main(args=None):
	rclpy.init(args=args)

	critic = Critic()

	rclpy.spin(critic)

	# Destroy the node explicitly
	# (optional - otherwise it will be done automatically
	# when the garbage collector destroys the node object)
	critic.destroy_node()
	rclpy.shutdown()


if __name__ == '__main__':
	main()