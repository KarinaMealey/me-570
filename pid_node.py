import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from ackermann_msgs.msg import AckermannDriveStamped, AckermannDrive
from rclpy.qos import qos_profile_sensor_data  # Quality of Service settings for real-time data
import time
from ros2_numpy import pose_to_np, to_ackermann
from collections import deque
import numpy as np

class PIDcontroller(Node):
    def __init__(self):
        super().__init__('pid_controller')

        # Define Quality of Service (QoS) for communication
        qos_profile = qos_profile_sensor_data  # Suitable for sensor data
        qos_profile.depth = 1  # Keep only the latest message

        # Create a publisher for sending AckermannDriveStamped messages to the '/autonomous/ackermann_cmd' topic
        self.publisher = self.create_publisher(AckermannDriveStamped, '/autonomous/ackermann_cmd', qos_profile)

        # Create a subscription to listen for PoseStamped messages from the '/waypoint' topic
        # When a message is received, the 'self.waypoint_callback' function is called
        self.subscription = self.create_subscription(
            PoseStamped,
            '/waypoint',
            self.waypoint_callback,
            qos_profile
        )

        # Load parameters
        self.params_set = False
        self.declare_params()
        self.load_params()

        # Create a timer that calls self.load_params every 10 seconds (10.0 seconds)
        self.timer = self.create_timer(10.0, self.load_params)

        self.last_time = time.time()
        self.last_steering_angle = 0.0

        # Initialize deque with a fixed length of self.max_out
        # This could be useful to allow the vehicle to temporarily lose the track for up to max_out frames before deciding to stop. (Currently not used yet.)
        self.max_out = 9
        self.success = deque([True] * self.max_out, maxlen=self.max_out)
        self.last_steer_error = 0.0
        self.last_ACC_error = 0.0
        self.ACC_error_acc = 0.0
        self.rel_speed = 0.0
        self.last_distance = 0.0
        self.front_car_speed = 0.0
        self.last_time = time.time()


    def waypoint_callback(self, msg: PoseStamped):

        # Convert incoming pose message to position, heading, and timestamp
        point, heading, timestamp_unix = pose_to_np(msg)

        # If the detected point contains NaN (tracking lost) stop the vehicle
        if np.isnan(point).any():
            ackermann_msg = to_ackermann(0.0, self.last_steering_angle, timestamp_unix)
            self.publisher.publish(ackermann_msg) # BRAKE
            self.success.append(False)
            return
        else:
            self.success.append(True)


        # Calculate time difference since last callback
        dt = timestamp_unix - self.last_time
        # Update the last time to the current timestamp
        self.last_time = timestamp_unix

        # Get x and y coordinates (ignore z), and compute the error in y
        x, y, z = point
        steer_error = -y
        ACC_error = self.desired_distance - x

        # Calculate the derivative of the error (change in error over time)
        d_steer_error = (steer_error-self.last_steer_error)/dt
        self.last_steer_error = steer_error

        # Compute the steering angle using a PD controller
        steering_angle = self.kp_steer*steer_error + self.kd_steer*d_steer_error

        """NEW PIDF LOOP FOR POSITION CONTROL"""
        # calculate error change for derivative control
        d_ACC_error = (ACC_error - self.last_ACC_error)/dt
        self.last_ACC_error = ACC_error
        self.speed = self.kp_ACC*ACC_error + self.ki_ACC*self.ACC_error_acc + self.kd_ACC*d_ACC_error + self.front_car_speed
        
        self.speed = max(min(self.max_speed, self.speed), 0.0)

        # Calculating Feed-forward term for the next iteration
        self.rel_speed = (x - self.last_distance)/dt
        self.last_distance = x
        #self.front_car_speed = self.rel_speed + self.speed

        # Get the timestamp from the message header
        timestamp = msg.header.stamp

        # Create an Ackermann drive message with speed and steering angle
        ackermann_msg = to_ackermann(self.speed, steering_angle, timestamp)

        # Publish the message to the vehicle
        self.publisher.publish(ackermann_msg)

        # Save the current error for use in the next iteration
        self.last_steer_error = steer_error

        self.get_logger().info(f"{self.speed=}")


    def declare_params(self):

        # Declare parameters with default values
        self.declare_parameters(
            namespace='',
            parameters=[
                ('kp_steer', 0.9),
                ('kd_steer', 0.1),
                ('kp_ACC', 0.9),
                ('ki_ACC', 0.0),
                ('kd_ACC', 0.0),
                ('max_speed', 1.5),
                ('desired_distance', 0.5),
                ('speed', 0.6),
            ]
        )

    def load_params(self):
        try:
            self.kp_steer = self.get_parameter('kp_steer').get_parameter_value().double_value
            self.kd_steer = self.get_parameter('kd_steer').get_parameter_value().double_value
            self.kp_ACC = self.get_parameter('kp_ACC').get_parameter_value().double_value
            self.ki_ACC = self.get_parameter('ki_ACC').get_parameter_value().double_value
            self.kd_ACC = self.get_parameter('kd_ACC').get_parameter_value().double_value
            self.max_speed = self.get_parameter('max_speed').get_parameter_value().double_value
            self.desired_distance = self.get_parameter('max_speed').get_parameter_value().double_value
            self.speed = self.get_parameter('speed').get_parameter_value().double_value

            if not self.params_set:
                self.get_logger().info("Parameters loaded successfully")
                self.params_set = True

        except Exception as e:
            self.get_logger().error(f"Failed to load parameters: {e}")

def main(args=None):
    rclpy.init(args=args)

    node = PIDcontroller()
    rclpy.spin(node)

    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
