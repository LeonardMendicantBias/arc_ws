import os

import rclpy
import rclpy.node
from ament_index_python.packages import get_package_share_directory

import numpy as np

from sensor_msgs.msg import Image
from arc_interfaces.msg import Code

from dall_e import unmap_pixels, load_model
from dall_e import Decoder

import torch
import torch.nn.functional as F

VOCAB_SIZE = 8192
BITS_PER_CODEWORD = 13  # 2^13 == 8192


class CodeSubscriber(rclpy.node.Node):

	def __init__(self):
		super().__init__('code_subscriber')
		self.img_width, self.img_height = 640, 480

		self.code_sub = self.create_subscription(
			Code,
			'/camera/camera/color/code',
			self.code_callback, 1
		)
		self.rec_pub = self.create_publisher(Image, '/camera/camera/color/reconstructed', 1)

		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		# self.device = "cpu"

		share_dir = get_package_share_directory('client')
		self.dec: Decoder = load_model(os.path.join(share_dir, 'checkpoints', 'decoder.pkl'))
		for param in self.dec.parameters():
			param.requires_grad_(False)
		self.dec.eval()
		self.dec.to(self.device)

	def code_callback(self, msg: Code):
		now_ns = self.get_clock().now().nanoseconds

		# Unpack flat uint8 bytes back into (N,) codeword indices.
		# Publisher packed N*13 bits MSB-first. To recover each 13-bit value, pad 3 zero
		# bits at the front of each row (not the end) to form a 16-bit big-endian word.
		bits = np.unpackbits(np.frombuffer(bytes(msg.data), dtype=np.uint8))
		n_codewords = len(bits) // msg.length
		codeword_bits = bits[:n_codewords * msg.length].reshape(n_codewords, msg.length)  # (N, 13)
		padded = np.zeros((n_codewords, 16), dtype=np.uint8)
		padded[:, 16 - msg.length:] = codeword_bits  # zero-pad at front → (N, 16)
		z_flat = np.packbits(padded, axis=1).flatten().view(np.dtype('>u2')).astype(np.int64)  # (N,)

		# The DALL-E encoder downsamples by 8x in each spatial dimension.
		h_prime = self.img_height // 8
		w_prime = self.img_width // 8
		z = torch.from_numpy(z_flat.reshape(1, h_prime, w_prime)).to(self.device)  # (1, H', W')

		z_one_hot = F.one_hot(z, num_classes=self.dec.vocab_size).permute(0, 3, 1, 2).float()
		with torch.no_grad(), torch.autocast(device_type=self.device.type, enabled=self.device.type == 'cuda'):
			x_stats = self.dec(z_one_hot).float()  # (1, 6, H, W)
		x_rec = unmap_pixels(torch.sigmoid(x_stats[:, :3]))  # (1, 3, H, W) in [0, 1]
		img_np = (x_rec * 255).clamp(0, 255).byte().squeeze(0).permute(1, 2, 0).cpu().numpy()

		rec_msg = Image()
		rec_msg.header.stamp = self.get_clock().now().to_msg()
		rec_msg.header.frame_id = msg.header.frame_id
		rec_msg.height = img_np.shape[0]
		rec_msg.width = img_np.shape[1]
		rec_msg.encoding = 'rgb8'
		rec_msg.is_bigendian = False
		rec_msg.step = img_np.shape[1] * 3
		rec_msg.data = img_np.tobytes()
		self.rec_pub.publish(rec_msg)

		latency_ms = (self.get_clock().now().nanoseconds - now_ns) / 1e6
		self.get_logger().info(f'reconstructed {img_np.shape[1]}x{img_np.shape[0]}  latency: {latency_ms:.4f} ms')


def main(args=None):
	rclpy.init(args=args)

	sub = CodeSubscriber()

	rclpy.spin(sub)

	sub.destroy_node()
	rclpy.shutdown()


if __name__ == '__main__':
	main()
