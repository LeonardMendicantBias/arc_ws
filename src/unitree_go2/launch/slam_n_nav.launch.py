
import os
from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    package_name = "unitree_go2"
    pkg_share = get_package_share_directory(package_name)

    # This file brings up ONLY the SLAM + Nav2 software stack. The robot itself
    # (robot_state_publisher, joint/lidar/odom publishers -- i.e. the sensors and
    # the TF tree) is launched separately by robot.launch.py. We rely on that
    # bring-up for the map -> odom -> base_link -> radar chain: unitree_go2_base
    # owns odom -> base_link and robot_state_publisher owns base_link -> radar;
    # slam_toolbox closes the loop with map -> odom.
    #
    # 2D SLAM (slam_toolbox) consumes a LaserScan on /scan, so the 3D lidar cloud
    # (/robot0/lidar/points from robot.launch.py) is flattened by
    # pointcloud_to_laserscan -- the one piece of input adaptation that is SLAM's
    # concern rather than the robot's, so it lives here.
    slam_params = os.path.join(pkg_share, "config",
                               "mapper_params_online_async.yaml")
    # Nav2 reuses that same SLAM-built map -> odom -> base_link -> radar chain,
    # so no AMCL/map_server is launched: slam_toolbox is the map source.
    nav2_params = os.path.join(pkg_share, "config", "nav2_params.yaml")

    use_sim_time = LaunchConfiguration("use_sim_time")

    return LaunchDescription([
        DeclareLaunchArgument(
            "use_sim_time",
            default_value="false",
            description="Use simulation (Gazebo) clock if true"
        ),
        DeclareLaunchArgument(
            "use_nav",
            default_value="true",
            description="Bring up the Nav2 navigation stack alongside SLAM"
        ),
        # Flatten the 3D cloud into a LaserScan on /scan for slam_toolbox.
        Node(
            package='pointcloud_to_laserscan',
            executable='pointcloud_to_laserscan_node',
            name='pointcloud_to_laserscan',
            output='screen',
            remappings=[('cloud_in', '/robot0/lidar/points'),
                        ('scan', '/scan')],
            parameters=[{
                'target_frame': 'radar',
                'transform_tolerance': 0.05,
                'min_height': -0.2,
                'max_height': 0.3,
                'angle_min': -3.14159,
                'angle_max': 3.14159,
                'angle_increment': 0.0087,   # ~0.5 deg
                'scan_time': 0.1,
                'range_min': 0.2,
                'range_max': 30.0,
                'use_inf': True,
            }],
        ),
        # 2D online async SLAM.
        Node(
            package='slam_toolbox',
            executable='async_slam_toolbox_node',
            name='slam_toolbox',
            output='screen',
            parameters=[slam_params,
                        {'use_sim_time': use_sim_time}],
        ),
        # Nav2 navigation stack (controller, planner, smoother, behaviors,
        # bt_navigator, velocity_smoother + lifecycle manager). AMCL/map_server
        # are intentionally skipped -- slam_toolbox supplies map -> odom live.
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(PathJoinSubstitution([
                FindPackageShare('nav2_bringup'), 'launch', 'navigation_launch.py'
            ])),
            condition=IfCondition(LaunchConfiguration("use_nav")),
            launch_arguments={
                'use_sim_time': use_sim_time,
                'params_file': nav2_params,
                'autostart': 'true',
            }.items(),
        ),
    ])
