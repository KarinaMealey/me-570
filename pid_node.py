import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from ackermann_msgs.msg import AckermannDriveStamped, AckermannDrive
from rclpy.qos import qos_profile_sensor_data  # Quality of Service settings for real-time data
import time
from ros2_numpy import pose_to_np, to_ackermann
import math

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

        self.last_steer_error = 0.0
        self.last_ACC_error = 0.0
        self.ACC_error_acc = 0.0
        self.last_time = time.time()

    def waypoint_callback(self, msg: PoseStamped):

        # Convert incoming pose message to position, heading, and timestamp
        point, heading, timestamp_unix = pose_to_np(msg)

        if any(math.isnan(x) for x in point):
            timestamp = msg.header.stamp
            ackermann_msg = to_ackermann(0.0, 0.0, timestamp)
            self.publisher.publish(ackermann_msg)
            return

        # Calculate time difference since last callback
        dt = timestamp_unix - self.last_time
        # Update the last time to the current timestamp
        self.last_time = timestamp_unix

        # Get x and y coordinates (ignore z), and compute the error in y
        x, y, z = point
        steer_error = -y
        ACC_error = self.desired_distance - z

        # Calculate the derivative of the error (change in error over time)
        d_steer_error = (steer_error-self.last_steer_error)/dt
        self.last_steer_error = steer_error

        # Compute the steering angle using a PD controller
        steering_angle = self.kp_steer*steer_error + self.kd_steer*d_steer_error

        """NEW PIDF LOOP FOR POSITION CONTROL"""
        d_ACC_error = (ACC_error - self.last_ACC_error)/dt
        self.last_ACC_error = ACC_error
        speed = self.kp_ACC*ACC_error + self.ki_ACC*self.ACC_error_acc + self.kd_ACC*d_ACC_error + 0.6
        self.ACC_error_acc += ACC_error*dt
        if speed > self.max_speed:
            speed = self.max_speed
        if speed < 0.0:
            speed = 0.0
        # Get the timestamp from the message header
        timestamp = msg.header.stamp

        # Create an Ackermann drive message with speed and steering angle
        ackermann_msg = to_ackermann(speed, steering_angle, timestamp)

        # Publish the message to the vehicle
        self.publisher.publish(ackermann_msg)

        # Save the current error for use in the next iteration
        self.last_steer_error = steer_error

        self.get_logger().info(f"{steering_angle=}")


    def declare_params(self):

        # Declare parameters with default values
        self.declare_parameters(
            namespace='',
            parameters=[
                ('kp_steer', 0.9),
                ('kd_steer', 0.0),
                ('kp_ACC', 0.9),
                ('ki_ACC', 0.1),
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
