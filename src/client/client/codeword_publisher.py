import rclpy
import rclpy.node
import rclpy.parameter
from rclpy.node import AsyncParametersClient

from std_msgs.msg import Header
from sensor_msgs.msg import Image, CompressedImage
from arc_interfaces.msg import Code

from dall_e import map_pixels, unmap_pixels, load_model
from dall_e import Encoder, Decoder


class CodePublisher(rclpy.node.Node):
	def __init__(self):
		super().__init__('code_publisher')

		self.declare_parameter('image_width', -1)
		self.declare_parameter('image_height', -1)

		self._param_client = AsyncParametersClient(self, '/camera/camera')
		self._resolution_timer = self.create_timer(0.5, self._fetch_camera_resolution)

		self.img_sub = self.create_subscription(
			Image,
			'/camera/camera/color/image_raw',
			self.img_callback, 1
		)

		self.mask_sub = self.create_subscription(
			Image,
			'/camera/camera/color/mask',
			self.mask_callback, 1
		)

		self.publisher_ = self.create_publisher(
			Code,
			'/camera/camera/color/code',
			1
		)

		self.timer = self.create_timer(1, self.timer_callback)

	async def _fetch_camera_resolution(self):
		if not self._param_client.service_is_ready():
			return

		result = await self._param_client.get_parameters(['rgb_camera.color_profile'])
		profile = result[0].string_value  # e.g. "1280x720x30"
		width, height, _ = (int(v) for v in profile.split('x'))

		self.set_parameters([
			rclpy.parameter.Parameter('image_width', rclpy.parameter.Parameter.Type.INTEGER, width),
			rclpy.parameter.Parameter('image_height', rclpy.parameter.Parameter.Type.INTEGER, height),
		])
		self.get_logger().info(f'Camera resolution: {width}x{height}')
		self._resolution_timer.cancel()

	def img_callback(self, img: Image):
		width = self.get_parameter('image_width').value
		height = self.get_parameter('image_height').value

	def mask_callback(self, mask: Image):
		pass
