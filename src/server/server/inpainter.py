import os
import sys
from ament_index_python.packages import get_package_share_directory

from typing import List
import yaml

import rclpy
from rclpy.node import Node

import numpy as np
import PIL.Image

from std_msgs.msg import Header
from sensor_msgs.msg import Image
from arc_interfaces.msg import DecodedImage, Mask

sys.path.insert(0, '/home/leonard/arc_ws/src')
from agentic.src.agents.tools.inpainting import InpaintingTool


class Inpainter(Node):

	def __init__(self):
		super().__init__('interpreter')

		self.img_width, self.img_height = 320, 240
		self.h_prime = self.img_height // 8
		self.w_prime = self.img_width // 8
		self.n_codewords = self.h_prime * self.w_prime

		self.recon_img_sub = self.create_subscription(
			DecodedImage, '/camera/camera/color/recon',
			self.img_callback, 1
		)
		self.inpaint_pub = self.create_publisher(Image, '/camera/camera/color/inpaint', 1)
		self.mask_pub = self.create_publisher(Image, '/camera/camera/color/abc', 1)

		share_dir = get_package_share_directory('server')
		self.inpainting_tool = InpaintingTool(
			ckpt=os.path.join(share_dir, 'checkpoints', 'dstt.pth'),
			refine_window=4, refine_length=16,
			online_window=7, n_neighbors=8,
			ref_step=20, compile_online=False
		)
		self.inpainting_tool.setup()
		self.inpainting_tool.reset()

	def img_callback(self, img: DecodedImage):
		now_ns = self.get_clock().now().nanoseconds

		img_np = np.frombuffer(img.data, dtype=np.uint8).reshape(img.height, img.width, 3)
		pil_img = PIL.Image.fromarray(img_np, 'RGB')
		orig_size = pil_img.size  # (W, H)

		mask_bits = np.array(img.mask, dtype=np.bool_)
		mask_grid = mask_bits.reshape(self.h_prime, self.w_prime).astype(np.uint8)
		mask_full = np.repeat(np.repeat(mask_grid, 8, axis=0), 8, axis=1)
		mask = PIL.Image.fromarray(mask_full * 255, 'L')
		# white: keep, black: inpaint

		# grid = (np.random.random((240 // 8, 320 // 8)) < 0.2).astype(np.uint8)
		# rand_mask = PIL.Image.fromarray(grid.repeat(8, axis=0).repeat(8, axis=1)*255, mode="L")

		mask_msg = Image()
		mask_msg.header.stamp = self.get_clock().now().to_msg()
		mask_msg.header.frame_id = img.header.frame_id
		mask_msg.height = mask_full.shape[0]
		mask_msg.width = mask_full.shape[1]
		mask_msg.encoding = 'rgb8'
		mask_msg.is_bigendian = False
		mask_msg.step = mask_full.shape[1] * 3
		mask_msg.data = mask.convert("RGB").tobytes()
		self.mask_pub.publish(mask_msg)

		# print(pil_img.size, mask.size, np.array(mask).min(), np.array(mask).max())
		result = self.inpainting_tool(
			PIL.Image.composite(pil_img, PIL.Image.new("RGB", mask.size, 0), mask),
			# pil_img,
			mask
		)

		result_np = np.asarray(result.resize(orig_size), dtype=np.uint8)
		out_msg = Image()
		out_msg.header.stamp = self.get_clock().now().to_msg()
		out_msg.header.frame_id = img.header.frame_id
		out_msg.height = result_np.shape[0]
		out_msg.width = result_np.shape[1]
		out_msg.encoding = 'rgb8'
		out_msg.is_bigendian = False
		out_msg.step = result_np.shape[1] * 3
		out_msg.data = result_np.tobytes()
		self.inpaint_pub.publish(out_msg)

		latency_ms = (self.get_clock().now().nanoseconds - now_ns) / 1e6
		self.get_logger().info(f'inpainted {out_msg.width}x{out_msg.height}  latency: {latency_ms:.4f} ms')
	
	def publish_inpaint(self, img: PIL.Image):
		result_np = np.asarray(img, dtype=np.uint8)
		out_msg = Image()
		out_msg.header.stamp = self.get_clock().now().to_msg()
		out_msg.header.frame_id = img.header.frame_id
		out_msg.height = result_np.shape[0]
		out_msg.width = result_np.shape[1]
		out_msg.encoding = 'rgb8'
		out_msg.is_bigendian = False
		out_msg.step = result_np.shape[1] * 3
		out_msg.data = result_np.tobytes()
		self.inpaint_pub.publish(out_msg)

def main(args=None):
	rclpy.init(args=args)

	inpainter = Inpainter()

	rclpy.spin(inpainter)

	# Destroy the node explicitly
	# (optional - otherwise it will be done automatically
	# when the garbage collector destroys the node object)
	inpainter.destroy_node()
	rclpy.shutdown()


if __name__ == '__main__':
	main()
