import math
import numpy as np

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from tf_transformations import euler_from_quaternion
import tf2_ros
from geometry_msgs.msg import TransformStamped

# Goal coordinates (x, y, theta) in the odom frame
GOAL = np.array([4.0, 10.0, -1.0])  # meters, meters, radians

ATTRACTIVE_K = 1.0  # Attraction strength towards the goal
REPULSIVE_K = 0.5  # Repulsive strength from obstacles
REPULSIVE_RADIUS = 1.2  # Threshold distance for obstacle detection

MAX_LINEAR = 0.6  # maximum forward velocity [m/s]
MAX_ANGULAR = 1.2  # maximum rotational velocity [rad/s]
GOAL_TOLERANCE = 0.15  # How close to the goal is considered reached


class PotentialFieldController(Node):
    def __init__(self):
        super().__init__('potential_field_controller')

        # Initialize transforms
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.create_subscription(LaserScan, '/scan', self.laser_callback, 10)
        self.create_subscription(Odometry, '/odom', self.odom_callback, 10)

        self.current_pose = None  # (x, y, yaw)

        self.timer = self.create_timer(0.1, self.control_loop)

        self.laser_data = None

    def odom_callback(self, msg):
        # Store pose in the odom frame
        position = msg.pose.pose.position
        orientation = msg.pose.pose.orientation
        _, _, yaw = euler_from_quaternion([orientation.x, orientation.y, orientation.z, orientation.w])
        self.current_pose = np.array([position.x, position.y, yaw])

    def laser_callback(self, msg):
        self.laser_data = msg

    def compute_attractive(self):
        # Direction to the goal in the odom frame
        dx = GOAL[0] - self.current_pose[0]
        dy = GOAL[1] - self.current_pose[1]
        attr = np.array([dx, dy])
        norm = np.linalg.norm(attr)
        if norm == 0:
            return np.zeros(2)
        return ATTRACTIVE_K * attr / norm

    def compute_repulsive(self):
        # Convert scan points into obstacle vectors
        if self.laser_data is None:
            return np.zeros(2)
        angles = np.arange(self.laser_data.angle_min, self.laser_data.angle_max, self.laser_data.angle_increment)
        rep_force = np.zeros(2)
        for r, theta in zip(self.laser_data.ranges, angles):
            if r < REPULSIVE_RADIUS and r > self.laser_data.range_min:
                # Point in base_link frame
                ox = r * np.cos(theta)
                oy = r * np.sin(theta)
                obs_vec = np.array([ox, oy])
                dist = np.linalg.norm(obs_vec)
                if dist == 0:
                    continue
                direction = -obs_vec / dist  # Direction away from the obstacle
                repulse = REPULSIVE_K * (1.0 / r - 1.0 / REPULSIVE_RADIUS) * (1.0 / (r ** 2)) * direction
                rep_force += repulse
        return rep_force

    def goal_reached(self):
        pos_diff = np.linalg.norm(GOAL[:2] - self.current_pose[:2])
        return pos_diff < GOAL_TOLERANCE

    def control_loop(self):
        if self.current_pose is None:
            return
        if self.goal_reached():
            self.cmd_pub.publish(Twist())  # Stop
            self.get_logger().info('Goal reached!')
            return

        # Compute attractive + repulsive field (in base_link frame!)
        attractive = self.compute_attractive()
        repulsive = self.compute_repulsive()
        total = attractive + repulsive

        # Total vector in robot orientation (base_link)
        yaw = self.current_pose[2]
        world2robot = np.array([[np.cos(-yaw), -np.sin(-yaw)],
                                [np.sin(-yaw), np.cos(-yaw)]])
        total_bl = world2robot @ total

        # Generate control commands
        cmd = Twist()
        cmd.linear.x = np.clip(total_bl[0], -MAX_LINEAR, MAX_LINEAR)
        cmd.angular.z = np.clip(total_bl[1], -MAX_ANGULAR, MAX_ANGULAR)
        self.cmd_pub.publish(cmd)


def main(args=None):
    rclpy.init(args=args)
    node = PotentialFieldController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
