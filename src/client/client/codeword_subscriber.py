import os

import rclpy
import rclpy.node
from ament_index_python.packages import get_package_share_directory

import numpy as np

from sensor_msgs.msg import Image
from arc_interfaces.msg import Code

from dall_e import map_pixels, unmap_pixels, load_model
from dall_e import Encoder, Decoder

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

		self.enc: Encoder = load_model(os.path.join(share_dir, 'checkpoints', 'encoder.pkl'))
		for param in self.enc.parameters():
			param.requires_grad_(False)
		self.enc.eval()
		self.enc.to(self.device)

		inp_mask = map_pixels(
			torch.zeros(1, 3, self.img_height, self.img_width, dtype=torch.float32, device=self.device)
		)
		with torch.no_grad():
			mask_logits = self.enc(inp_mask)  # (1, VOCAB_SIZE, H', W')
		# Pre-compute flat codeword indices for the fallback (non-transmitted) pixels
		self.mask_code_flat = (
			torch.argmax(mask_logits, dim=1).flatten().cpu().numpy().astype(np.int64)
		)  # (N,)

	def code_callback(self, msg: Code):
		now_ns = self.get_clock().now().nanoseconds

		h_prime = self.img_height // 8
		w_prime = self.img_width // 8
		n_total = h_prime * w_prime

		# Unpack the M transmitted codewords from packed bits.
		# msg.length == M (number of selected codewords); each is BITS_PER_CODEWORD bits wide.
		n_selected = msg.length  # M
		raw_bits = np.unpackbits(np.frombuffer(bytes(msg.data), dtype=np.uint8))
		codeword_bits = raw_bits[:n_selected * BITS_PER_CODEWORD].reshape(n_selected, BITS_PER_CODEWORD)  # (M, 13)
		padded = np.zeros((n_selected, 16), dtype=np.uint8)
		padded[:, 16 - BITS_PER_CODEWORD:] = codeword_bits  # zero-pad MSBs → (M, 16)
		z_selected = np.packbits(padded, axis=1).flatten().view(np.dtype('>u2')).astype(np.int64)  # (M,)

		# Build full z_flat (N,): seed with pre-computed mask fallback, fill transmitted positions.
		mask_bool = np.array(msg.mask, dtype=bool)  # (N,)
		z_flat = self.mask_code_flat.copy()          # (N,) fallback for non-transmitted pixels
		z_flat[mask_bool] = z_selected               # overwrite transmitted positions

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
