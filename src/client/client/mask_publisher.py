import rclpy
import rclpy.node

import numpy as np

from arc_interfaces.msg import Mask


class MaskPublisher(rclpy.node.Node):

    def __init__(self):
        super().__init__('mask_publisher')

        self.img_width, self.img_height = 320, 240
        self.h_prime = self.img_height // 8
        self.w_prime = self.img_width // 8
        self.n_codewords = self.h_prime * self.w_prime

        self.mask_pub = self.create_publisher(Mask, '/camera/camera/color/mask', 1)
        self.create_timer(1.0, self.timer_callback)

    def timer_callback(self):
        mask = np.random.randint(0, 2, size=self.n_codewords, dtype=np.bool_)
        # mask = np.ones(self.n_codewords, dtype=np.bool_)

        msg = Mask()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.mask = mask.tolist()
        self.mask_pub.publish(msg)

        n_selected = int(mask.sum())
        self.get_logger().info(f'published mask: {n_selected}/{self.n_codewords} selected')


def main(args=None):
    rclpy.init(args=args)

    pub = MaskPublisher()

    rclpy.spin(pub)

    pub.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
