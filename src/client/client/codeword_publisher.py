import os

import rclpy
import rclpy.node
from ament_index_python.packages import get_package_share_directory

import numpy as np

from sensor_msgs.msg import Image
from arc_interfaces.msg import Code, Mask

import cv2
import torch

from dall_e import map_pixels, load_model
from dall_e import Encoder

VOCAB_SIZE = 8192
BITS_PER_CODEWORD = 13  # 2^13 == 8192


class CodePublisher(rclpy.node.Node):

	def __init__(self):
		super().__init__('code_publisher')

		self.img_width, self.img_height = 320, 240
		self.h_prime = self.img_height // 8
		self.w_prime = self.img_width // 8
		self.n_codewords = self.h_prime * self.w_prime

		share_dir = get_package_share_directory('client')
		self.img_sub = self.create_subscription(
			Image, '/camera/camera/color/image_raw',
			self.img_callback, 1
		)
		self.mask_sub = self.create_subscription(
			Mask, '/camera/camera/color/mask',
			self.mask_callback, 1
		)
		self.code_pub = self.create_publisher(Code, '/camera/camera/color/code', 1)

		self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

		self.enc: Encoder = load_model(os.path.join(share_dir, 'checkpoints', 'encoder.pkl'), self.device)
		for param in self.enc.parameters():
			param.requires_grad_(False)
		self.enc.eval()

		# Warm-up pass to compute mask codewords
		self.mask = np.ones(self.h_prime*self.w_prime, dtype=np.bool_)
		print("type", self.device.type)

	def _enc(self, x: torch.Tensor) -> torch.Tensor:
		with torch.autocast(device_type=self.device.type, enabled=self.device.type=='cuda'):
			return self.enc(x).float()
		
	def mask_callback(self, msg: Mask):
		self.mask = np.frombuffer(msg.mask, dtype=np.bool_)#.reshape(msg.height, msg.width, -1)

	def img_callback(self, msg: Image):
		now_ns = self.get_clock().now().nanoseconds

		# Skip PIL: convert ROS image bytes directly to a float tensor in [0, 1]
		img_np = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, -1)
		if img_np.shape[0] != self.img_height or img_np.shape[1] != self.img_width:
			img_np = cv2.resize(img_np, (self.img_width, self.img_height), interpolation=cv2.INTER_LINEAR)
		inp_frame = (
			torch.from_numpy(img_np[:, :, :3].copy())
			.permute(2, 0, 1).float().div_(255).unsqueeze(0).to(self.device)
		)
		inp_frame = map_pixels(inp_frame)  # (1, 3, H, W)

		z_logits = self._enc(inp_frame)
		z = torch.argmax(z_logits, dim=1)  # (1, H', W')

		# Pack each codeword as 13 bits into a flat uint8 byte array.
		# View each uint16 value as big-endian 2 bytes, unpack to 16 bits,
		# drop the top 3 zero bits, then repack the N*13 bits into bytes.
		z_flat = z.squeeze(0).flatten().cpu().numpy().astype(np.uint16)  # (N,)

		# Select only codewords where mask == 1
		mask_bool = self.mask.astype(bool)
		z_selected = z_flat[mask_bool]  # (M,)
		n_selected = len(z_selected)

		if n_selected == 0:
			return

		z_bytes = np.frombuffer(z_selected.astype('>u2').tobytes(), dtype=np.uint8)  # (M*2,)
		bits = np.unpackbits(z_bytes).reshape(n_selected, 16)[:, 16 - BITS_PER_CODEWORD:]  # (M, 13)
		packed = np.packbits(bits.flatten()).tobytes()

		code_msg = Code()
		code_msg.header.stamp = self.get_clock().now().to_msg()
		code_msg.header.frame_id = msg.header.frame_id
		code_msg.length = n_selected
		code_msg.data = packed
		code_msg.mask = self.mask.tolist()
		self.code_pub.publish(code_msg)

		latency_ms = (self.get_clock().now().nanoseconds - now_ns) / 1e6
		self.get_logger().info(f'codewords: {n_selected}/{self.n_codewords}  packed: {len(packed)} bytes  latency: {latency_ms:.4f} ms')


def main(args=None):
	rclpy.init(args=args)

	pub = CodePublisher()

	rclpy.spin(pub)

	pub.destroy_node()
	rclpy.shutdown()


if __name__ == '__main__':
	main()
