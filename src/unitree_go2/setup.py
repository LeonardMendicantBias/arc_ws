import os
from glob import glob

from setuptools import find_packages, setup

package_name = 'unitree_go2'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob(os.path.join('launch', '*launch.[pxy][yma]*'))),
        (os.path.join('share', package_name, 'urdf'), glob(os.path.join('urdf', '*.urdf'))),
        (os.path.join('share', package_name, 'dae'), glob(os.path.join('dae', '*'))),
        (os.path.join('share', package_name, 'meshes'), glob(os.path.join('meshes', '*'))),
        (os.path.join('share', package_name, 'config'), glob(os.path.join('config', '*'))),
        (os.path.join('share', package_name, 'calibration'), glob(os.path.join('calibration', '*'))),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Leonard ngo',
    maintainer_email='ngoak@islab.snu.ac.kr',
    description='This package brings up the necessary component of the Unitree Go2 Edu robot dog',
    license='Apache-2.0',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            "camera_publisher = unitree_go2.front_camera_publisher:main",
            "joint_state_publisher = unitree_go2.lowstate_joint_publisher:main",
            "lidar_publisher = unitree_go2.lidar_publisher:main",
            "base_node = unitree_go2.unitree_go2_base:main",
        ],
    },
)
