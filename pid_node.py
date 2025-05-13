import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from ackermann_msgs.msg import AckermannDriveStamped, AckermannDrive
from rclpy.qos import qos_profile_sensor_data  # Quality of Service settings for real-time data
import time
from ros2_numpy import pose_to_np, to_ackermann
from collections import deque
import numpy as np
import time

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
        self.way_sub = self.create_subscription(PoseStamped,'/waypoint', self.waypoint_callback, qos_profile)
        self.obj_sub = self.create_subscription(PoseStamped,'/object', self.obj_callback, qos_profile)

        # Load parameters
        self.params_set = False
        self.declare_params()
        self.load_params()

        # Create a timer that calls self.load_params every 10 seconds (10.0 seconds)
        self.timer = self.create_timer(10.0, self.load_params)

        self.last_error = 0.0
        self.last_time = time.time()
        self.last_steering_angle = 0.0

        # Initialize deque with a fixed length of self.max_out
        # This could be useful to allow the vehicle to temporarily lose the track for up to max_out frames before deciding to stop. (Currently not used yet.)
        self.max_out = 9
        self.success = deque([True] * self.max_out, maxlen=self.max_out)

        self.len_history = 10
        self.id2target = {0: 'car', 1: 'center'}

        # Initialize target metadata for each label (e.g. stop sign, speed signs)
        self.targets = {
            lbl: {
                'id': id,  # Numerical class ID
                'history': deque([False] * self.len_history, maxlen=self.len_history),  # Detection history
                'detected': False,   # Whether the target is currently detected
                'reacted': False,    # Whether the vehicle has already reacted
                'threshold': 0.5,    # Detection threshold
                'distance': 0.0,     # Current distance to the target
                'min_distance': 2.0  # Distance threshold to start reacting
            }
            for id, lbl in self.id2target.items()
}
    def stop_and_wait(self, target, duration=2.0):
        # Update detection flag based on average detection history
        target['detected'] = True if np.mean(target['history']) > target['threshold'] else False

        # Only stop if target is detected, not yet reacted, and close enough
        if target['detected'] and not target['reacted']:
            if target['distance'] < target['min_distance']:
                ackermann_msg = to_ackermann(0.0, self.last_steering_angle)
                self.publisher.publish(ackermann_msg)           # BRAKE
                self.get_logger().info("Brake for 2 sec")
                time.sleep(duration)                            # Wait for a specified duration
                target['reacted'] = True                        # Mark target as handled

        if not any(target['history']) and target['reacted']:
            target['reacted'] = False  # Reset reaction flag if the target hasn't been detected in any recent frames


    def obj_callback(self, msg: PoseStamped):

        # Convert incoming pose message to position, heading, and timestamp
        point, heading, timestamp_unix = pose_to_np(msg)

        id = int(point[-1]) # Extract class ID stored in the z-coordinate
        x, _ = point[:2] # Get distance
        lbl = self.id2target[id]
        data = self.targets[lbl]
        self.get_logger().info(f"{lbl=}")

        
        # Check if point is NaN (object not detected or not visible)
        if np.isnan(point).any():
            data['history'].append(False)
            #self.get_logger().info("Object Not Detected")
        else:
            data['history'].append(True)
            data['distance'] = x
            #self.get_logger().info("Object Detected")

        # Check if the model has detected the car
        # If so, decrease speed appropriately
        if lbl == 'car':
            x = max(self.min_dist, x)
            self.speed = (1-self.min_dist/x)*self.max_speed
            self.get_logger().info(f"{self.speed=}")
            


    def waypoint_callback(self, msg: PoseStamped):

        # Convert incoming pose message to position, heading, and timestamp
        point, heading, timestamp_unix = pose_to_np(msg)

        # If the detected point contains NaN (tracking lost) stop the vehicle
        if np.isnan(point).any():
            self.success.append(False)
            if any(self.success):
                # Keep driving with the last known steering angle unless the line is lost for self.max_out consecutive frames
                ackermann_msg = to_ackermann(self.speed, self.last_steering_angle, timestamp_unix)
                self.get_logger().info("Lost Line, Following Prev Commands")
                self.get_logger().info(f"{self.speed=}")
            else:
                ackermann_msg = to_ackermann(0.0, self.last_steering_angle, timestamp_unix)
                self.publisher.publish(ackermann_msg) # BRAKE
            return
        else:
            self.success.append(True)


        # Calculate time difference since last callback
        dt = timestamp_unix - self.last_time

        # Update the last time to the current timestamp
        self.last_time = timestamp_unix

        # Get x and y coordinates (ignore z), and compute the error in y
        x, y, _ = point
        error = 0.0 - y  # Assuming the goal is to stay centered at y = 0

        # Calculate the derivative of the error (change in error over time)
        d_error = (error - self.last_error) / dt 

        # Compute the steering angle using a PD controller
        steering_angle = (self.kp * error) + (self.kd * d_error)

        # Get the timestamp from the message header
        timestamp = msg.header.stamp

        # Create an Ackermann drive message with speed and steering angle
        ackermann_msg = to_ackermann(self.speed, steering_angle, timestamp)

        # Publish the message to the vehicle
        self.publisher.publish(ackermann_msg)

        # Save the current error for use in the next iteration
        self.last_error = error
        self.get_logger().info(f"{error=}")

        # If tracking is lost, continue driving with the last known steering angle
        # to follow the previously estimated path (e.g., maintain the current curve)
        self.last_steering_angle = steering_angle
        global angle 
        angle = steering_angle


    def declare_params(self):

        # Declare parameters with default values
        self.declare_parameters(
            namespace='',
            parameters=[
                ('kp', 0.9),  # OG Number=9
                ('kd', 0.0),  # OG Number=0
                ('speed', 0.6),
                ('max_speed', 1.5),
                ('min_dist', 2.0)
            ]
        )

    def load_params(self):
        try:
            self.kp = self.get_parameter('kp').get_parameter_value().double_value
            self.kd = self.get_parameter('kd').get_parameter_value().double_value
            self.speed = self.get_parameter('speed').get_parameter_value().double_value
            self.min_dist = self.get_parameter('min_dist').get_parameter_value().double_value
            self.max_speed = self.get_parameter('max_speed').get_parameter_value().double_value

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
