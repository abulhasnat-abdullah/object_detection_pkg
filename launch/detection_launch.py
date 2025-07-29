from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import TimerAction, Shutdown

def generate_launch_description():
    # Both nodes
    camera_node = Node(
        package='object_detection_pkg',
        executable='rosbot_camera_subscriber',
        name='rosbot_camera_subscriber',
        output='screen'
    )

    detection_node = Node(
        package='object_detection_pkg',
        executable='object_detection_node',
        name='object_detection_node',
        output='screen'
    )
    return LaunchDescription([
        camera_node,
        detection_node,
        
    ])