import os
from typing import List
from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, Command, PathJoinSubstitution
from launch_ros.actions import Node
from launch.actions import IncludeLaunchDescription, DeclareLaunchArgument
from launch.launch_description_sources import FrontendLaunchDescriptionSource, PythonLaunchDescriptionSource

from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    package_name = "unitree_go2"  # Replace with your package name

    return LaunchDescription([
        DeclareLaunchArgument(
            "use_sim_time",
            default_value="false",
            description="Use simulation (Gazebo) clock if true"
        ),
        DeclareLaunchArgument(
            'robot_description_file',
            default_value='go2_description.urdf',
            description='URDF file for the robot'
        ),
        # robot_state_publisher is a stock ROS node (not a domain bridge), so it
        # obeys ROS_DOMAIN_ID. It consumes /joint_states and publishes
        # /robot_description + /tf, all of which must live on the OUTPUT domain
        # (1) where SLAM/Nav2/RViz consume them -- hence the explicit override.
        # The bridged joint publisher below feeds it /joint_states on domain 1.
        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            name='go2_robot_state_publisher',
            output='screen',
            additional_env={'ROS_DOMAIN_ID': '1'},
            parameters=[{
                # 'use_sim_time': use_sim_time,
                'robot_description': ParameterValue(Command(['xacro ', PathJoinSubstitution([
                    FindPackageShare(package_name),
                    'urdf',
                    LaunchConfiguration('robot_description_file')
                ])]), value_type=str)
            }],
            # arguments=[self.config.config_paths['urdf']]
        ),
        # Bridges the robot's live LowState (DDS) onto /joint_states so the
        # RViz model tracks the real robot. Replaces the static
        # joint_state_publisher, which only emitted constant default angles.
        Node(
            package='unitree_go2',
            executable='joint_state_publisher',
            name='lowstate_joint_publisher',
            output='screen',
        ),
        # Republishes the Go2 utlidar cloud (utlidar/cloud) as a clean,
        # SLAM-ready PointCloud2 on /robot0/lidar/points, re-framed onto the
        # URDF 'radar' link and re-stamped with the local clock.
        Node(
            package='unitree_go2',
            executable='lidar_publisher',
            name='lidar_publisher',
            output='screen',
        ),
        Node(
            package='unitree_go2',
            executable='camera_publisher',
            name='camera_publisher',
            output='screen',
        ),
        # Publishes the Go2 sport odometry as the odom -> base_link TF (and a
        # matching /odom) so the map -> odom -> base_link -> sensor TF chain
        # that SLAM/Nav2 require is complete.
        Node(
            package='unitree_go2',
            executable='base_node',
            name='unitree_go2_base',
            output='screen',
        ),
    ])