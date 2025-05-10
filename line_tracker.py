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
from line_follower.utils import get_corners, to_surface_coordinates, read_transform_config, draw_box, parse_predictions, get_base, detect_bbox_center
from ros2_numpy import image_to_np, np_to_image, np_to_pose, np_to_compressedimage

class LineFollower(Node):
    def __init__(self, package_dir, filepath, debug = False):
        super().__init__('line_tracker')
        self.pkg_dir = package_dir
        self.index=0
        # Define a message to send when the line tracker has lost track
        self.lost_msg = PoseStamped()
        self.lost_msg.pose.position.x = self.lost_msg.pose.position.y = self.lost_msg.pose.position.z = float('nan')

        # Plot the result if debug is True
        self.debug = debug

        if not os.path.exists(filepath):
            raise FileNotFoundError(f"The file at '{filepath}' was not found.")

        # Read the homography matrix H from the given config file.
        # This matrix defines the transformation from 2D pixel coordinates to 3D world coordinates.
        H = read_transform_config(filepath)

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
        
        # Publisher to send processed result images for visualization
        if self.debug:
            self.im_publisher = self.create_publisher(CompressedImage, '/result', qos_profile)


        # Load parameters
        self.params_set = False
        self.declare_params()
        self.load_params()

        # Create a timer that calls self.load_params every 10 seconds (10.0 seconds)
        self.timer = self.create_timer(10.0, self.load_params)

        # Returns a function that converts pixel coordinates to surface coordinates using a fixed matrix 'H'
        self.to_surface_coordinates = lambda u, v: to_surface_coordinates(u, v, H)

        model_path = package_dir + '/models/best.pt'
        self.model = YOLO(model_path)
        self.prev_x = 0
        self.prev_y = 0
        self.get_logger().info("Line Tracker Node started.")
        self.get_logger().info(f"names = {self.model.names}")
        


    def image_callback(self, msg):
        # Convert ROS image to numpy format
        # timestamp_unix is the image timestamp in seconds (Unix time)
        image, timestamp_unix = image_to_np(msg)

        # get the predictions of the current image based on the model trained on line images
        predictions = self.model(image, verbose=False)

        car_success, point1, point2 = detect_bbox_center(predictions,0)
        self.get_logger().info(f"{car_success=}")
        plot = predictions[0].plot()

        if car_success:
            x, _ = self.to_surface_coordinates(point1, point2)
            #self.get_logger().info("Sending x!")
        else:
            x = self.prev_x
            #cv2.imwrite(f"{self.pkg_dir}/output/sample_{self.index}.png", image)
            self.index+=1
        
        #line_success, mask = parse_predictions(predictions, class_ids=[0])
        line_success, mask = parse_predictions(predictions, class_ids=[1])
        if line_success:
            cx, cy = get_base(mask)
            _, y = self.to_surface_coordinates(cx, cy)
        else:
            y = self.prev_y

        # Convert back to ROS2 Image and publish
        if car_success or line_success:
            #cv2.circle(plot, point, 10, color=(255, 0, 0), thickness=cv2.FILLED)
            timestamp = msg.header.stamp
            
            # convert the world coords to a pose and publish it
            pose_msg = np_to_pose(np.array([x, y]), 0.0, timestamp=timestamp)
            self.publisher.publish(pose_msg)

        else: 
            self.publisher.publish(self.lost_msg)
            #self.get_logger().info("Lost track!")
        
        im_msg = np_to_compressedimage(cv2.cvtColor(plot, cv2.COLOR_BGR2RGB))
        
        # Publish
        self.im_publisher.publish(im_msg)
        """# Convert ROS image to numpy format
        # timestamp_unix is the image timestamp in seconds (Unix time)
        image, timestamp_unix = image_to_np(msg)

        # Apply preprocessing (Canny edge detection)
        im_canny = self.preprocess_image(image)
        
        # Find the line center position
        success, self.win_x = self.find_center(self.win_x, im_canny)

        # Plot and publish result if debug is Trze
        if self.debug:
            corners = self.get_corners(self.win_x)
            result = draw_box(image, im_canny, corners)
            im_msg = np_to_image(result)
            self.im_publisher.publish(im_msg)

        if success:
            
            # Transform from pixel to world coordinates
            x, y = self.to_surface_coordinates(self.win_x, self.win_y)

            # Extract the timestamp from the incoming image message
            timestamp = msg.header.stamp

            # Publish waypoint as Pose message
            pose_msg = np_to_pose(np.array([x, y]), 0.0, timestamp=timestamp)
            self.publisher.publish(pose_msg)

        else:
            self.publisher.publish(self.lost_msg)
            self.get_logger().info("Lost track!")"""

    def preprocess_image(self, image):

        # Convert to grayscale
        im_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Apply Gaussian blur to the grayscale image
        im_blur = cv2.GaussianBlur(im_gray, self.kernel, 0)

        # Apply Canny edge detection to the preprocessed image
        im_canny = cv2.Canny(im_blur, self.canny_min, self.canny_max) # [0 or 255]
        
        return im_canny

    def find_center(self, win_x, im_canny):
        # Adjust window center by detecting lane line pixels in the image patch
        # Success means pixels were found and a new center is calculated

        x1, x2, y1, y2 = self.get_corners(win_x)

        # Extract the relevant image patch and identify non-zero pixels (lane line pixels)
        patch = im_canny[y1:y2, x1:x2]
        _, xs = np.nonzero(patch)


        # Check if any lane line pixels are detected
        # If not, return False along with the current center
        if len(xs) == 0:
            return False, win_x

        # Calculate the new center based on the mean of the detected pixels
        updated_center = int(x1 + np.mean(xs))

        return True, updated_center

    def declare_params(self):

        # Declare parameters with default values
        self.declare_parameters(
            namespace='',
            parameters=[
                ('win_h', 20),
                ('win_w', 90),
                ('win_x', 310),
                ('win_y', 280),
                ('image_w', 640),
                ('image_h', 360),
                ('canny_min', 80),
                ('canny_max', 180),
                ('k', 3)
            ]
        )

    def load_params(self):
        try:
            self.win_h = self.get_parameter('win_h').get_parameter_value().integer_value
            self.win_w = self.get_parameter('win_w').get_parameter_value().integer_value
            self.win_x = self.get_parameter('win_x').get_parameter_value().integer_value
            self.win_y = self.get_parameter('win_y').get_parameter_value().integer_value
            self.image_w = self.get_parameter('image_w').get_parameter_value().integer_value
            self.image_h = self.get_parameter('image_h').get_parameter_value().integer_value
            self.canny_min = self.get_parameter('canny_min').get_parameter_value().integer_value
            self.canny_max = self.get_parameter('canny_max').get_parameter_value().integer_value
            self.k = self.get_parameter('k').get_parameter_value().integer_value

            # Ensure kernel is at least 3 and an odd number
            self.k = max(3, self.k + (self.k % 2 == 0))
            self.kernel = (self.k, self.k)

            # Returns a function that calculates corner points with fixed window and image parameters
            self.get_corners = lambda win_x: get_corners(win_x, self.win_y, self.win_w, self.win_h, self.image_w, self.image_h)

            if not self.params_set:
                self.get_logger().info("Parameters loaded successfully")
                self.params_set = True

        except Exception as e:
            self.get_logger().error(f"Failed to load parameters: {e}")

def main(args=None):

    # Transformation matrix for converting pixel coordinates to world coordinates
    pkg_dir = get_package_prefix('line_follower').replace('install', 'src')
    filepath = get_package_share_directory('line_follower') + '/config/transform_config_640x360.yaml'
            
    rclpy.init(args=args)
    node = LineFollower(pkg_dir, filepath, debug = True)
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()