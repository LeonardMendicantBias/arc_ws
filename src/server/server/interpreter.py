from typing import List
import yaml

import rclpy
from rclpy.node import Node

import numpy as np

from std_msgs.msg import Header
from rcl_interfaces.msg import Parameter, ParameterValue, ParameterType, SetParametersResult
from rcl_interfaces.srv import SetParameters

from smolagents import CodeAgent, HfApiModel, FinalAnswerTool, GoogleSearchTool, VisitWebpageTool


class Interpreter(Node):

	def __init__(self):
		super().__init__('interpreter')

		self.declare_parameter('application_name', '')

		self._vision_expert_client = self.create_client(
			SetParameters, '/vision_expert/set_parameters'
		)
		self._critic_client = self.create_client(
			SetParameters, '/critic/set_parameters'
		)

		self.add_on_set_parameters_callback(self._on_parameters_changed)

		model = HfApiModel(
			max_tokens=4906,
			temperature=1.0,
			model_id="meta-llama/Meta-Llama-3-8B-Instruct", # it is possible that this model may be overloaded
			custom_role_conversions=None,
		)

		with open("/home/nvidia/arc_ws/src/agentic/src/agents/interpreter.yaml", 'r') as stream:
			prompt_templates = yaml.safe_load(stream)

		web_search = GoogleSearchTool()
		visit_webpage = VisitWebpageTool()
		final_answer = FinalAnswerTool()
		# final_answer 
		self.agent = CodeAgent(
			model=model,
			tools=[
				web_search,
				visit_webpage,
				final_answer,
			],
			max_steps=6,
			verbosity_level=2,
			grammar=None,
			planning_interval=None,
			name=None,
			description=None,
			prompt_templates=prompt_templates,
		)

	def _on_parameters_changed(self, params):
		for param in params:
			if param.name == 'application_name':
				self.get_logger().info(f"application_name updated: '{param.value}'")

				result = self.agent.run(param.value)
				description, modalities, obj_classes, metric = result['DESCRIPTION'], result['MODALITIES'], result['CLASSES'], result['METRIC']

				self._push_to_vision_expert(description, modalities, obj_classes)
				self._push_to_critic(metric)

		return SetParametersResult(successful=True)

	def _push_to_vision_expert(self, description: str, modalities: List[str], classes: List[str]):
		req = SetParameters.Request()
		req.parameters = [
			Parameter(name='description', value=ParameterValue(
				type=ParameterType.PARAMETER_STRING,
				string_value=description)),
			Parameter(name='modalities', value=ParameterValue(
				type=ParameterType.PARAMETER_STRING_ARRAY,
				string_array_value=modalities)),
			Parameter(name='classes', value=ParameterValue(
				type=ParameterType.PARAMETER_STRING_ARRAY,
				string_array_value=classes)),
		]
		future = self._vision_expert_client.call_async(req)
		future.add_done_callback(self._on_param_set)

	def _push_to_critic(self, metric: str):
		req = SetParameters.Request()
		req.parameters = [
			Parameter(name='metric', value=ParameterValue(
				type=ParameterType.PARAMETER_STRING,
				string_value=metric)),
		]
		future = self._critic_client.call_async(req)
		future.add_done_callback(self._on_param_set)

	def _on_param_set(self, future):
		try:
			results = future.result()
			for r in results.results:
				if not r.successful:
					self.get_logger().warn(f"Parameter set failed: {r.reason}")
		except Exception as e:
			self.get_logger().error(f"Parameter set exception: {e}")


def main(args=None):
	rclpy.init(args=args)

	interpreter = Interpreter()

	rclpy.spin(interpreter)

	# Destroy the node explicitly
	# (optional - otherwise it will be done automatically
	# when the garbage collector destroys the node object)
	interpreter.destroy_node()
	rclpy.shutdown()


if __name__ == '__main__':
	main()
