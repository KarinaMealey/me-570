# Third-Party Libraries
import os
import cv2
import yaml
import numpy as np
from ultralytics import YOLO

# ROS Imports
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data  # Quality of Service settings for real-time data
from sensor_msgs.msg import Image, CompressedImage
from geometry_msgs.msg import PoseStamped
from ament_index_python.packages import get_package_share_directory, get_package_prefix

# Project-Specific Imports
from line_follower.utils import to_surface_coordinates, read_transform_config, parse_predictions, get_base, detect_bbox_center
from ros2_numpy import image_to_np, np_to_image, np_to_pose, np_to_compressedimage

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
        self.publisher = self.create_publisher(PoseStamped, 'waypoint', qos_profile)

        # Publisher to send 3d object positions
        self.obj_publisher = self.create_publisher(PoseStamped, 'object', qos_profile)
        
        # Publisher to send processed result images for visualization
        self.im_publisher = self.create_publisher(CompressedImage, 'result', qos_profile)

        # Returns a function that converts pixel coordinates to surface coordinates using a fixed matrix 'H'
        self.to_surface_coordinates = lambda u, v: to_surface_coordinates(u, v, H)

        # Load the custom trained YOLO model
        self.model = self.load_model(model_path)

        # Map class IDs to labels and labels to IDs
        id2label = self.model.names
        #self.get_logger.info(f"Class names = {id2label}")
        targets = ['car', 'center']
        self.id2target = {id: lbl for id, lbl in id2label.items() if lbl in targets}
        self.prev_x = 0
        self.prev_y = 0
        # Log an informational message indicating that the Line Tracker Node has started
        self.get_logger().info("Line Tracker Node started. Custom YOLO model loaded successfully.")

    def load_model(self, filepath):
        model = YOLO(filepath)

        self.imgsz = model.args['imgsz'] # Get the image size (imgsz) the loaded model was trained on.

        # Init model
        print("Initializing the model with a dummy input...")
        im = np.zeros((self.imgsz, self.imgsz, 3)) # dummy image
        _ = model.predict(im)  
        print("Model initialization complete.")

        return model

    def image_callback(self, msg):

       # Convert ROS image to numpy format
        # timestamp_unix is the image timestamp in seconds (Unix time)
        image, timestamp_unix = image_to_np(msg)

        # get the predictions of the current image based on the model trained on line images
        predictions = self.model(image, verbose=False)

        car_success, point1, point2 = detect_bbox_center(predictions,0)
        plot = predictions[0].plot()

        if car_success:
            x, _ = self.to_surface_coordinates(point1, point2)
        else:
            x = self.prev_x
        
        #line_success, mask = parse_predictions(predictions, class_ids=[0])
        line_success, mask = parse_predictions(predictions, class_ids=[1])
        if line_success:
            cx, cy = get_base(mask)
            _, y = self.to_surface_coordinates(cx, cy)
        else:
            y = self.prev_y

        # Convert back to ROS2 Image and publish
        if car_success or line_success:
            timestamp = msg.header.stamp
            
            # convert the world coords to a pose and publish it
            pose_msg = np_to_pose(np.array([x, y]), 0.0, timestamp=timestamp)
            self.publisher.publish(pose_msg)

        else: 
            self.publisher.publish(self.stop_msg)
            #self.get_logger().info("Lost track!")

        for id, lbl in self.id2target.items():
            detected, u, v = detect_bbox_center(predictions, id)

            if detected:

                # Transform from pixel to world coordinates
                x, y = self.to_surface_coordinates(u, v)

                # Publish object as Pose message
                pose_msg = np_to_pose(np.array([x, y, id]), 0.0, timestamp=timestamp)
            else:
                pose_msg = self.stop_msg # Attention: this creates a reference — use deepcopy() if you want self.stop_msg to remain unchanged
                pose_msg.pose.position.z = float(id)

            self.obj_publisher.publish(pose_msg)



def main(args=None):

    # Transformation matrix for converting pixel coordinates to world coordinates
    config_path = get_package_share_directory('line_follower') + '/config/transform_config_640x360.yaml'

    # Path to your custom trained YOLO model
    pkg_path = get_package_prefix('line_follower').replace('install', 'src') # /mxck2_ws/install/line_follower → /mxck2_ws/src/line_follower
    model_path = pkg_path + '/models/best.pt'
            
    rclpy.init(args=args)
    node = LineFollower(model_path, config_path)
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()