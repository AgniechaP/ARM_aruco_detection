#!/usr/bin/env python3

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, DeclareLaunchArgument
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node as LaunchNode
import numpy as np


def generate_launch_description():
    os.environ['TURTLEBOT3_MODEL'] = 'waffle'
    os.environ['GAZEBO_MODEL_PATH'] = '/arm_ws/src/arm05_sim/models'
    launch_file_dir = os.path.join(get_package_share_directory('turtlebot3_gazebo'), 'launch')
    nav2_launch_file_dir = os.path.join(get_package_share_directory('turtlebot3_navigation2'), 'launch')
    print(nav2_launch_file_dir)
    pkg_gazebo_ros = get_package_share_directory('gazebo_ros')

    use_sim_time = LaunchConfiguration('use_sim_time', default='true')
    x_pose = LaunchConfiguration('x_pose', default='0.0')
    y_pose = LaunchConfiguration('y_pose', default='0.0')
    aruco_seed = LaunchConfiguration('aruco_seed', default='')

    aruco_seed_launch_arg = DeclareLaunchArgument(
        'aruco_seed',
        default_value=str(np.random.randint(0, 9999999)),
        description="Seed for aruco spawner"
    )

    world = os.path.join(
        get_package_share_directory('arm05_sim'),
        'worlds',
        'arm_house.world'
    )

    gzserver_cmd = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_gazebo_ros, 'launch', 'gzserver.launch.py')
        ),
        launch_arguments={'world': world}.items()
    )

    gzclient_cmd = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_gazebo_ros, 'launch', 'gzclient.launch.py')
        )
    )

    robot_state_publisher_cmd = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(launch_file_dir, 'robot_state_publisher.launch.py')
        ),
        launch_arguments={'use_sim_time': use_sim_time}.items()
    )

    spawn_turtlebot_cmd = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(launch_file_dir, 'spawn_turtlebot3.launch.py')
        ),
        launch_arguments={
            'x_pose': x_pose,
            'y_pose': y_pose
        }.items()
    )
    print(os.path.join(nav2_launch_file_dir, 'navigation2.launch.py'))
    navigation_cmd = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(nav2_launch_file_dir, 'navigation2.launch.py')
        ),
        launch_arguments={
            'map': "/arm_ws/src/arm05_sim/map/map.yaml",
            'params_file': "/arm_ws/src/arm05_sim/param/waffle.yaml"
        }.items()
    )

    spawn_aruco_cubes = LaunchNode(
        namespace="spawn_cubes", package="arm05_sim", executable="spawn_aruco.py", output="screen", arguments=['--seed', aruco_seed]
    )

    ld = LaunchDescription()

    # Add the commands to the launch description
    ld.add_action(aruco_seed_launch_arg)
    ld.add_action(gzserver_cmd)
    ld.add_action(gzclient_cmd)
    ld.add_action(robot_state_publisher_cmd)
    ld.add_action(spawn_aruco_cubes)
    ld.add_action(spawn_turtlebot_cmd)
    ld.add_action(navigation_cmd)
    

    return ld

