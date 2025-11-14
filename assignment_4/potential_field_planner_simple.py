import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
import tf_transformations
import tf2_ros
import math

class PotentialFieldPlanner(Node):
    def __init__(self):
        super().__init__('potential_field_planner')

        # subscribers and publishers
        self.scan_sub = self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.odom_sub = self.create_subscription(Odometry, '/odom', self.odom_callback, 10)

        # tf buffer and listener
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # goal
        self.goal = [4.0, 10.0]

        # params
        self.k_attr = 0.8
        self.k_rep = 0.4
        self.rep_thresh = 1.2
        self.max_lin = 0.6
        self.max_ang = 1.0

        # state
        self.robot_x = 0.0
        self.robot_y = 0.0
        self.robot_yaw = 0.0

    def odom_callback(self, msg):
        # read robot pose
        self.robot_x = msg.pose.pose.position.x
        self.robot_y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        _, _, yaw = tf_transformations.euler_from_quaternion([q.x, q.y, q.z, q.w])
        self.robot_yaw = yaw

    def scan_callback(self, msg):
        # compute attractive velocity
        dx = self.goal[0] - self.robot_x
        dy = self.goal[1] - self.robot_y
        dist_to_goal = math.sqrt(dx*dx + dy*dy)

        if dist_to_goal > 0.05:
            ux = dx / dist_to_goal
            uy = dy / dist_to_goal
        else:
            ux, uy = 0.0, 0.0

        v_attr_x = self.k_attr * ux
        v_attr_y = self.k_attr * uy

        # compute repulsive velocity from lidar
        v_rep_x = 0.0
        v_rep_y = 0.0

        angle = msg.angle_min
        for r in msg.ranges:
            if 0.05 < r < self.rep_thresh:
                ox = r * math.cos(angle)
                oy = r * math.sin(angle)

                mag = (1.0/r - 1.0/self.rep_thresh)
                scale = self.k_rep * mag / (r*r)

                v_rep_x += scale * (-ox / r)
                v_rep_y += scale * (-oy / r)

            angle += msg.angle_increment

        # combine fields
        vx = v_attr_x + v_rep_x
        vy = v_attr_y + v_rep_y

        # convert velocity into robot frame
        robot_heading = self.robot_yaw
        vx_r = math.cos(robot_heading)*vx + math.sin(robot_heading)*vy
        vy_r = -math.sin(robot_heading)*vx + math.cos(robot_heading)*vy

        # compute command
        cmd = Twist()
        cmd.linear.x = max(min(vx_r, self.max_lin), -self.max_lin)
        cmd.angular.z = max(min(2.0 * math.atan2(vy_r, vx_r), self.max_ang), -self.max_ang)

        # stop if extremely close
        if dist_to_goal < 0.25:
            cmd.linear.x = 0.0
            cmd.angular.z = 0.0

        self.cmd_pub.publish(cmd)


def main(args=None):
    rclpy.init(args=args)
    node = PotentialFieldPlanner()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
