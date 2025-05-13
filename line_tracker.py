# Third-Party Libraries
import os
import cv2
import numpy as np

# ROS Imports
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data  # Quality of Service settings for real-time data
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseStamped
from ament_index_python.packages import get_package_share_directory, get_package_prefix


# Project-Specific Imports
from line_follower.utils import to_surface_coordinates, read_transform_config, parse_predictions, get_base, detect_bbox_center, draw_circle
from ros2_numpy import image_to_np, np_to_image, np_to_pose, np_to_image
from yolo_onnx_runner import YOLO # pip install git+https://github.com/william-mx/yolo-onnx-runner.git

class LineFollower(Node):
    def __init__(self, model_path, config_path):
        super().__init__('line_tracker')

        # Define a message to send when the line tracker has lost track
        self.stop_msg = PoseStamped()
        self.stop_msg.pose.position.x = self.stop_msg.pose.position.y = self.stop_msg.pose.position.z = float('nan')

        for path in [model_path, config_path]:
            if not os.path.exists(path):
                raise FileNotFoundError(f"The file at '{path}' was not found.")


        # Read the homography matrix H from the given config file.
        # This matrix defines the transformation from 2D pixel coordinates to 3D world coordinates.
        H = read_transform_config(config_path)

        # Define Quality of Service (QoS) for communication
        qos_profile = qos_profile_sensor_data  # Suitable for sensor data
        qos_profile.depth = 1  # Keep only the latest message

        # Subscriber to receive camera images
        self.im_subscriber = self.create_subscription(
            Image, 
            '/camera/camera/color/image_raw',  # Topic name
            self.image_callback,  # Callback function to process incoming images
            qos_profile
        )

        # Publisher to send calculated waypoints
        self.publisher = self.create_publisher(PoseStamped, '/waypoint', qos_profile)

        # Publisher to send 3d object positions
        self.obj_publisher = self.create_publisher(PoseStamped, '/object', qos_profile)
        
        # Publisher to send processed result images for visualization
        self.im_publisher = self.create_publisher(Image, '/result', qos_profile)

        # Returns a function that converts pixel coordinates to surface coordinates using a fixed matrix 'H'
        self.to_surface_coordinates = lambda u, v: to_surface_coordinates(u, v, H)

        # Load the custom trained YOLO model
        self.model = YOLO(model_path, conf_thres=0.1)

        # Map class IDs to labels and labels to IDs
        id2label = self.model.names
        targets = ['car', 'center'] # Classes to track and publish as /object PoseStamped messages
        self.id2target = {id: lbl for id, lbl in id2label.items() if lbl in targets}

        # Log an informational message indicating that the Line Tracker Node has started
        self.get_logger().info("Line Tracker Node started. Custom YOLO ONNX model loaded successfully.")


    def image_callback(self, msg):

        # Convert ROS image to numpy format
        # timestamp_unix is the image timestamp in seconds (Unix time)
        image, timestamp_unix = image_to_np(msg)


        # Run YOLO inference
        predictions = self.model(image)

        # Draw results on the image
        plot = predictions[0].plot()

        # Identify the next waypoint along the lane line
        success, mask = parse_predictions(predictions)
        


        if success:
            cx, cy = get_base(mask)

            # Transform from pixel to world coordinates
            x, y = self.to_surface_coordinates(cx, cy)

            # Extract the timestamp from the incoming image message
            timestamp = msg.header.stamp

            # Publish waypoint as Pose message
            pose_msg = np_to_pose(np.array([x, y]), 0.0, timestamp=timestamp)
            self.publisher.publish(pose_msg)
            self.get_logger().info("On track!")

        else:
            self.get_logger().info("Lost track!")
            self.publisher.publish(self.stop_msg)
            

        # Compute 3D positions of all targets (e.g., stop sign)
        for id, lbl in self.id2target.items():
            detected, u, v = detect_bbox_center(predictions, id)

            if detected:

                # Draw a circle at the bottom center of the bounding box
                plot = draw_circle(plot, u, v)

                # Transform from pixel to world coordinates
                x, y = self.to_surface_coordinates(u, v)

                # Publish object as Pose message
                pose_msg = np_to_pose(np.array([x, y, id]), 0.0, timestamp=timestamp)
            else:
                pose_msg = self.stop_msg # Attention: this creates a reference — use deepcopy() if you want self.stop_msg to remain unchanged
                pose_msg.pose.position.z = float(id)

            self.obj_publisher.publish(pose_msg)

        # Convert back to ROS2 Image and publish
        im_msg = np_to_image(cv2.cvtColor(plot, cv2.COLOR_BGR2RGB))

        # Publish predictions
        self.im_publisher.publish(im_msg)



def main(args=None):

    # Transformation matrix for converting pixel coordinates to world coordinates
    config_path = get_package_share_directory('line_follower') + '/config/transform_config_640x360.yaml'

    # Path to your custom trained YOLO model
    pkg_path = get_package_prefix('line_follower').replace('install', 'src') # /mxck2_ws/install/line_follower → /mxck2_ws/src/line_follower
    model_path = pkg_path + '/models/best.onnx'
            
    rclpy.init(args=args)
    node = LineFollower(model_path, config_path)
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()