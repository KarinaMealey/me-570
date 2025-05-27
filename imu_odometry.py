#!/usr/bin/env python3

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu
from nav_msgs.msg import Path
from std_srvs.srv import Empty
from rclpy.qos import qos_profile_sensor_data  # Quality of Service settings for real-time data
import time
import matplotlib.pyplot as plt
from ament_index_python.packages import get_package_prefix
import csv
from scipy.signal import butter, lfilter_zi, lfilter
from ros2_numpy import imu_to_np


class OdometryNode(Node):
    def __init__(self, pkg_dir, every_nth = 20):
        super().__init__('odometry')
        self.get_logger().info("Initializing IMU odometry node")

        self.output_dir = pkg_dir + '/results'

        # Create a service called 'reset_odometry' using the Empty service type.
        self.srv = self.create_service(Empty, 'reset_odometry', self.reset_odometry)

        # Only publish every nth waypoint
        self.every_nth = every_nth
    
        # Define Quality of Service (QoS) for communication
        qos_profile = qos_profile_sensor_data  # Suitable for sensor data
        qos_profile.depth = 1  # Keep only the latest message

        # Create a subscriber for the calibrated IMU data
        # Use /camera/imu for RealSense IMU
        self.subscription = self.create_subscription(Imu, '/imu', self.imu_callback, qos_profile)
        
        # Create a publisher for the resulting odometry path
        self.imu_pub = self.create_publisher(Path, 'path', 10)

        self.last_time = time.time()
        self.init_time = self.last_time
        # This correction applies only to the microcontroller IMU.
        # We selected a gyro range of ±500°/s but mistakenly used the sensitivity for ±250°/s (131 instead of 65.5).
        # This underestimates the angular velocity by a factor of 2, so we multiply the result by 2 to correct it.
        # For the RealSense IMU, no correction is needed — use yaw_gain = 1.0.
        self.yaw_gain = 2.0

        # These are the zero-offsets determined by averaging the car data at rest.
        self.rest_yaw_rate = -0.1201
        self.rest_accel_x = -0.52919


        """BUTTERWORTH FILTER STUFF"""
        # Setting up real-time butterworth filters by filtering in the z-domain
        self.b_accel, self.a_accel = butter(3, 0.01, btype='low')
        self.b_yaw, self.a_yaw = butter(3, 0.1, btype='low')
        self.zi_accel = lfilter_zi(self.b_accel, self.a_accel)
        self.zi_yaw_rate = lfilter_zi(self.b_yaw, self.a_yaw)

        # detertmine if this is the first data collected to set time to be zero
        self.first_it = True

        # Initialize the position of the car
        self.init_position()

    def init_position(self):
        self.path = []
        self.data = []
        self.speed = 0.0
        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0

    def reset_odometry(self, request, response):
        self.init_position()
        self.get_logger().info("Vehicle odometry reset to default")
        return response

    def imu_callback(self, msg: Imu):
        """
        Callback that processes IMU data to update odometry.
        """
        # Convert IMU message to numpy format
        # timestamp_unix is the image timestamp in seconds (Unix time)
        data, timestamp_unix = imu_to_np(msg) # [accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z]

        # Set the initial time for data collection
        if self.first_it:
            self.first_it = False
            self.init_time = timestamp_unix

        # Coordinate transformation for microcontroller IMU (accel_y → accel_x, gyro_z → yaw_rate)
        # For RealSense IMU (accel_z → accel_x, gyro_y → yaw_rate)
        accel_x = data[1]
        yaw_rate = data[-1] * self.yaw_gain

        # Data debugging of camera imu
        curr_norm_time = timestamp_unix-self.init_time
        self.data.append([data[0], data[1], data[2], data[3], data[4], data[5], curr_norm_time])

        # Filter data if needed (e.g., clipping, bias correction, or noise reduction)
        # using real time butterworth filter
        # Subtract out the offset for the yaw gain and accel
        accel_x = accel_x - self.rest_accel_x
        yaw_rate = yaw_rate - self.yaw_gain*self.rest_yaw_rate

        """BUTTERWORTH FILTER STUFF"""
        # Do real-time butterworth filtering for both the acceleration and the yaw rate
        accel_x_temp, self.zi_accel = lfilter(self.b_accel, self.a_accel, [accel_x], zi=self.zi_accel)
        yaw_rate_temp, self.zi_yaw_rate = lfilter(self.b_yaw, self.a_yaw, [yaw_rate], zi=self.zi_yaw_rate)
        accel_x = accel_x_temp[0]
        yaw_rate = yaw_rate_temp[0]

        self.get_logger().info(f"Got {accel_x=} and {yaw_rate=}")

        # Calculate time difference since last callback
        dt = timestamp_unix - self.last_time

        # Update speed and orientation
        self.speed += accel_x*dt
        self.theta += yaw_rate*dt

        # Update position
        self.x += np.cos(np.deg2rad(self.theta)) * self.speed * dt
        self.y += np.sin(np.deg2rad(self.theta)) * self.speed * dt

        self.path.append([self.x, self.y])

        self.last_time = timestamp_unix


    def plot_path(self, make_csv=False):
        # Convert path to numpy array for easier slicing
        path_array = np.array(self.path)  # shape (N, 2)

        # Plot
        plt.figure()
        plt.plot(path_array[:, 0], path_array[:, 1], marker='o', linewidth=2)
        plt.xlabel("X position")
        plt.ylabel("Y position")
        plt.title("IMU Odomentry Vehicle Path")
        plt.grid(True)
        plt.axis("equal")  # Keep aspect ratio

        # Save to file
        plt.savefig(self.output_dir + "/odometry_path.png")
        plt.close()  # Close the figure to free memory

        # Save data to a csv file if we want to
        if make_csv:
            filename = self.output_dir + '/data.csv'
            with open(filename, 'w', newline='') as csvfile:
                csvwriter = csv.writer(csvfile)
                csvwriter.writerows(self.data)


    def cleanup(self):

        # Plot result
        self.plot_path(make_csv=False)

        self.get_logger().info("Shutting down odometry node, cleaning up resources.")
        # Add any additional cleanup logic here, like saving logs or closing files


def main(args=None):
    pkg_dir = get_package_prefix('odometry').replace('install', 'src') # /mxck2_ws/install/odometry → /mxck2_ws/src/odometry

    rclpy.init(args=args)
    node = OdometryNode(pkg_dir)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Odometry node interrupted by user.")
    finally:
        node.cleanup()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
