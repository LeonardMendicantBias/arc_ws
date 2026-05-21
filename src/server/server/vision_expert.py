import re

import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter

import yaml
import numpy as np
import PIL

from std_msgs.msg import Header
from rcl_interfaces.msg import SetParametersResult
from sensor_msgs.msg import Image
from arc_interfaces.msg import Mask

from agents.utils import export_masks, FinalAnswerTool
from agents.tools.reid import ReIDTool
from agents.tools.inpainting import InpaintingTool
from agents.tools.segmentation import SemanticSegmentationTool
from agents.tools.transmission import EncodingTool

from smolagents import CodeAgent, HfApiModel
from smolagents import ActionStep

import torch


class VisionExpert(Node):

	def __init__(self):
		super().__init__('vision_expert')

		self.recon_img_sub = self.create_subscription(
			Image,
			'/camera/camera/color/reconstructed',
			self.img_callback, 1
		)
		self.mask_pub = self.create_publisher(Mask, '/camera/camera/color/mask', 1)

		self.declare_parameter('description', Parameter.Type.STRING)
		self.declare_parameter('modalities', Parameter.Type.STRING_ARRAY)
		self.declare_parameter('classes', Parameter.Type.STRING_ARRAY)

		self.add_on_set_parameters_callback(self._on_parameters_changed)

		model = HfApiModel(
			max_tokens=4906,
			temperature=0.5,
			model_id="meta-llama/Meta-Llama-3-8B-Instruct",
			custom_role_conversions=None,
		)

		with open("./vision.yaml", 'r') as stream:
			prompt_templates = yaml.safe_load(stream)

		self.inpainting_tool = InpaintingTool(
			ckpt='/home/nvidia/arc_ws/src/agentic/checkpoints/dstt.pth'
		)
		self.semantic_segmentation_tool = SemanticSegmentationTool(model_name="CIDAS/clipseg-rd64-refined")
		self.reid_tool = ReIDTool(0.5, model_name="kadirnar/osnet_x0_25_imagenet")
		self.final_answer = FinalAnswerTool()

		self.agent = CodeAgent(
			model=model,
			tools=[
				self.inpainting_tool,
				self.reid_tool,
				self.semantic_segmentation_tool,
				self.final_answer,
			],
			additional_authorized_imports=["matplotlib", "PIL", "typing"],
			max_steps=1,
			verbosity_level=2,
			grammar=None,
			planning_interval=None,
			name=None,
			description=None,
			prompt_templates=prompt_templates,
		)
		
		self.pipeline = None

	def _on_parameters_changed(self, params):
		desp, modalities, obj_classes = '', [], []
		for param in params:
			if param.name == 'description':
				self.get_logger().info(f"description updated: '{param.value}'")
				desp = param.value
			elif param.name == 'modalities':
				self.get_logger().info(f"modalities updated: {param.value}")
				modalities = param.value
			elif param.name == 'classes':
				self.get_logger().info(f"classes updated: {param.value}")
				obj_classes = param.value

		# Run vision expert ONCE on a sample to determine the processing pipeline
		sample_frame = PIL.Image.new("RGB", (432, 240))
		sample_mask_t = torch.ones(1, 1, 1, 240, 432)
		sample_mask = PIL.Image.fromarray(255 * sample_mask_t[0, 0, 0].numpy().astype(np.uint8), "L")

		self.agent.state.update({
			"rgb_frame": sample_frame,  # a placeholder
			"rgb_mask": sample_mask,  # a placeholder
			"depth_frame": sample_frame,  # a placeholder
			"depth_mask": sample_mask,  # a placeholder
			"object_classes": obj_classes,  # this from interpreter
		})
		interpreter_output = desp
		self.agent.run(interpreter_output)

		# Extract the generated code from the last ActionStep
		pipeline_code = next(
			step.tool_calls[0].arguments
			for step in reversed(self.agent.logs)
			if isinstance(step, ActionStep) and step.tool_calls
		)
		print("Generated pipeline:\n", pipeline_code)

		def build_pipeline(code_str, **tools):
			"""Compile agent-generated pipeline code into a reusable callable."""
			body = re.sub(r'\bfinal_answer\((.+)\)', r'__result__ = \1', code_str, flags=re.DOTALL)

			def pipeline(rgb_frame=None, rgb_mask=None, depth_frame=None, depth_mask=None, object_classes=None):
				ns = {
					**tools,
					"rgb_frame": rgb_frame,
					"rgb_mask": rgb_mask,
					"depth_frame": depth_frame,
					"depth_mask": depth_mask,
					"object_classes": object_classes,
					"__result__": None,
				}
				exec(body, ns)
				return ns["__result__"]

			return pipeline

		self.pipeline = build_pipeline(
			pipeline_code,
			inpainting_tool=self.inpainting_tool,
			semantic_segmentation_tool=self.semantic_segmentation_tool,
			reid_tool=self.reid_tool,
		)

		return SetParametersResult(successful=True)
	
	def img_callback(self, img: Image):
		pass

def main(args=None):
	rclpy.init(args=args)

	expert = VisionExpert()

	rclpy.spin(expert)

	# Destroy the node explicitly
	# (optional - otherwise it will be done automatically
	# when the garbage collector destroys the node object)
	expert.destroy_node()
	rclpy.shutdown()


if __name__ == '__main__':
	main()