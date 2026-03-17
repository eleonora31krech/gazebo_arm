#!/usr/bin/env python3

import os
from launch import LaunchDescription
from launch.actions import ExecuteProcess, RegisterEventHandler
from launch.event_handlers import OnProcessExit
from launch.substitutions import PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

from moveit_configs_utils import MoveItConfigsBuilder
def generate_launch_description():
    pkg_panda_gz = FindPackageShare('panda_gz_moveit')

    world_file = PathJoinSubstitution([
        pkg_panda_gz,
        'worlds',
        'panda_grasp.sdf'
    ])

    urdf_file = '/workspace/src/panda_gz_moveit/urdf/panda_arm.urdf'
    with open(urdf_file, 'r', encoding='utf-8') as f:
        robot_description_content = f.read()

    robot_description = {'robot_description': robot_description_content}


    gazebo_server = ExecuteProcess(
        cmd=['gz', 'sim', '-v4', '-r', '-s', world_file],
        output='screen',
    )

    gazebo_client = ExecuteProcess(
        cmd=['gz', 'sim', '-v4', '-g'],
        output='screen',
    )


    spawn_entity = Node(
        package='ros_gz_sim',
        executable='create',
        arguments=[
            '-topic', 'robot_description',
            '-name', 'panda',
            '-x', '0.20',
            '-y', '0.0',
            '-z', '0.85',
            #  (home position)
            '-J', 'panda_joint1', '0.0',
            '-J', 'panda_joint2', '-0',
            '-J', 'panda_joint3', '0.0',
            '-J', 'panda_joint4', '0',
            '-J', 'panda_joint5', '0.0',
            '-J', 'panda_joint6', '0',
            '-J', 'panda_joint7', '0.',
            '-J', 'panda_finger_joint1', '0.04',
            '-J', 'panda_finger_joint2', '0.04',
        ],
        output='screen',
    )

    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        output='screen',
        parameters=[robot_description],
    )


    joint_state_broadcaster = Node(
        package='controller_manager',
        executable='spawner',
        arguments=['joint_state_broadcaster'],
        output='screen',
    )

    arm_controller = Node(
        package='controller_manager',
        executable='spawner',
        arguments=['panda_arm_controller'],
        output='screen',
    )

    hand_controller = Node(
        package='controller_manager',
        executable='spawner',
        arguments=['panda_hand_controller'],
        output='screen',
    )


    # spawn_joint_state_broadcaster = RegisterEventHandler(
    #     event_handler=OnProcessExit(
    #         target_action=spawn_entity,
    #         on_exit=[joint_state_broadcaster],
    #     )
    # )

    # spawn_arm_controller = RegisterEventHandler(
    #     event_handler=OnProcessExit(
    #         target_action=joint_state_broadcaster,
    #         on_exit=[arm_controller],
    #     )
    # )

    spawn_hand_controller = RegisterEventHandler(
        event_handler=OnProcessExit(
            target_action=arm_controller,
            on_exit=[hand_controller],
        )
    )


    camera_bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        arguments=[
            '/camera/image@sensor_msgs/msg/Image@gz.msgs.Image',
            '/camera/depth_image@sensor_msgs/msg/Image@gz.msgs.Image',
            '/camera/camera_info@sensor_msgs/msg/CameraInfo@gz.msgs.CameraInfo',
            '/camera/points@sensor_msgs/msg/PointCloud2@gz.msgs.PointCloudPacked',
        ],
        output='screen',
    )


    camera_tf_publisher = Node(
        package='panda_gz_moveit',
        executable='camera_tf_publisher.py',
        name='camera_tf_publisher',
        output='screen',
    )


    color_detector = Node(
        package='panda_gz_moveit',
        executable='color_cube_detector.py',
        name='color_cube_detector',
        output='screen',
    )
    move_arm = Node(
        package='panda_gz_moveit',
        executable='pick_place_controller.py',
        name='pick_place_controller',
        output='screen',
    )


    rviz_config = PathJoinSubstitution([
        pkg_panda_gz,
        'config',
        'moveit.rviz'
    ])

    rviz = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', rviz_config],
        parameters=[robot_description],
        output='screen',
    )


    joint_bridge=Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        arguments=[
            '/panda_joint1_cmd@std_msgs/msg/Float64@gz.msgs.Double',
            '/panda_joint2_cmd@std_msgs/msg/Float64@gz.msgs.Double',
            '/panda_joint3_cmd@std_msgs/msg/Float64@gz.msgs.Double',
            '/panda_joint4_cmd@std_msgs/msg/Float64@gz.msgs.Double',
            '/panda_joint5_cmd@std_msgs/msg/Float64@gz.msgs.Double',
            '/panda_joint6_cmd@std_msgs/msg/Float64@gz.msgs.Double',
            '/panda_joint7_cmd@std_msgs/msg/Float64@gz.msgs.Double',

            '/panda_finger_joint1_cmd@std_msgs/msg/Float64@gz.msgs.Double',
            '/panda_finger_joint2_cmd@std_msgs/msg/Float64@gz.msgs.Double',
        ],
        output='screen'
    )

    # moveit_config = (
    #     MoveItConfigsBuilder("panda", package_name="panda_moveit_config")
    #     .robot_description(file_path=urdf_file)
    #     .to_moveit_configs()
    # )

    # move_group_node = Node(
    #     package="moveit_ros_move_group",
    #     executable="move_group",
    #     output="screen",
    #     parameters=[
    #         moveit_config.to_dict(),
    #         {"use_sim_time": True},
    #     ],
    # )

    # return LaunchDescription([
    #     gazebo_server,
    #     gazebo_client,
    #     robot_state_publisher,
    #     spawn_entity,
    #     # spawn_joint_state_broadcaster,
    #     # spawn_arm_controller,
    #     spawn_hand_controller,
    #     camera_bridge,
    #     camera_tf_publisher,
    #     joint_bridge,
    #     # color_detector,
    #     move_arm,
    #     rviz,
    # ])
    return LaunchDescription([
        gazebo_server,
        gazebo_client,
        robot_state_publisher,
        spawn_entity,
        # spawn_hand_controller,
        camera_bridge,
        camera_tf_publisher,
        joint_bridge,
        move_arm,
        # color_detector,
        # move_arm,
        # rviz,
    ])