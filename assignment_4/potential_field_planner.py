import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from rclpy.duration import Duration
from geometry_msgs.msg import Twist, PointStamped, TransformStamped
from sensor_msgs.msg import LaserScan
import tf2_ros
import tf2_geometry_msgs
from tf_transformations import euler_from_quaternion
import math
import threading
from rcl_interfaces.msg import ParameterDescriptor


class PotentialFieldPlanner(Node):
    def __init__(self):
        super().__init__('potential_field_planner')
        # best effort for consuming sensor data
        qos_sub = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            durability=DurabilityPolicy.VOLATILE
        )
        # reliability for sending messages to cmd_vel
        qos_pub = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            durability=DurabilityPolicy.VOLATILE
        )
        self.publisher = self.create_publisher(Twist, '/cmd_vel', qos_pub)
        self.subscription = self.create_subscription(LaserScan, '/scan', self.scan_callback, qos_sub)

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # declare parameters
        self.declare_parameter('k_a', 0.3, ParameterDescriptor(description='Attraction gain. Lower means less attraction. Higher means more aggressive attraction.'))
        self.declare_parameter('rho_0', 1.5, ParameterDescriptor(description='Repulsion threshold. Obstacles within range contribute to repulsion.'))
        self.declare_parameter('k_r', 1.0, ParameterDescriptor(description='Repulsion gain. Lower means less repulsion. Higher means more aggressive repulsion.'))
        self.declare_parameter('v_r_max', 1.0, ParameterDescriptor(description='Constant. Upper bound for repulsion. Higher values will be clamped.'))
        self.declare_parameter('k_ang', 1.0, ParameterDescriptor(description='Angular gain. Lower means less aggressive turns towards the goal direction. Higher means more aggressive turns.'))
        self.declare_parameter('v_max_linear', 0.5, ParameterDescriptor(description='Constant. Upper bound for desired linear velocity. Higher values will be clamped.'))
        self.declare_parameter('v_max_angular', 1.0, ParameterDescriptor(description='Constant. Upper bound for desired angular speed. Higher values will be clamped.'))
        self.declare_parameter('approach_threshold', 0.2, ParameterDescriptor(description='Distance threshold. When the goal is within range it is approached more deliberately.'))
        self.declare_parameter('ang_threshold', 0.1, ParameterDescriptor(description='Angular threshold. When the robot is more or less facing the goal no turning is required.'))

        # get parameter values
        self.k_a = self.get_parameter('k_a').value
        self.rho_0 = self.get_parameter('rho_0').value
        self.k_r = self.get_parameter('k_r').value
        self.v_r_max = self.get_parameter('v_r_max').value
        self.ang_threshold = self.get_parameter('ang_threshold').value
        self.k_ang = self.get_parameter('k_ang').value
        self.v_max_linear = self.get_parameter('v_max_linear').value
        self.v_max_angular = self.get_parameter('v_max_angular').value
        self.approach_threshold = self.get_parameter('approach_threshold').value

        # allow updating parameters during operation
        self.add_on_set_parameters_callback(self.params_callback)

        # static goal in odom
        self.goal_o = PointStamped()
        self.goal_o.header.frame_id = 'odom'
        self.goal_o.point.x = 4.0
        self.goal_o.point.y = 10.0
        self.goal_o.point.z = 0.0
        self.goal_theta_o = -1.0

        self.goal_reached = False

        # thread-safe scan buffer
        self.latest_scan = None
        self.lock = threading.Lock()

        # 10hz planning timer
        timer_period = 0.1
        self.timer = self.create_timer(timer_period, self.planning_callback)

        self.planning_counter = 0
        self.planning_near_goal_counter = 0

    def params_callback(self, params):
        for param in params:
            if param.name == 'k_a':
                self.k_a = param.value
            elif param.name == 'k_r':
                self.k_r = param.value
        return rclpy.node.SetParametersResult(successful=True)

    def scan_callback(self, msg):
        # lock for safe scan update
        with self.lock:
            self.latest_scan = msg

    def planning_callback(self):
        # access scan safely
        do_log = self.planning_counter % 10 == 0
        self.planning_counter += 1
        with self.lock:
            if self.latest_scan is None:
                return
            # freshness check
            scan_time = self.latest_scan.header.stamp
            current_time = self.get_clock().now()
            if current_time - rclpy.time.Time.from_msg(scan_time) > Duration(seconds=1.0):
                self.get_logger().warn("Stale scan detected! Stopping robot for now to avoid potential accidents.")
                twist = Twist()
                self.publisher.publish(twist)
                return
            # reference to immutable object instead of cloning is okay
            scan = self.latest_scan

        if self.goal_reached:
            twist = Twist()
            self.publisher.publish(twist)
            return

        # transform check
        if not self.tf_buffer.can_transform('base_link', 'odom', rclpy.time.Time()):
            self.get_logger().warn('Currently cannot transform from o to b!')
            return

        try:
            # lookup tf from odom to base_link via buffer subscribed to /tf topic
            # this provides transform to map goal from odom to base_link frame
            tf_o_to_b = self.tf_buffer.lookup_transform('base_link', 'odom', rclpy.time.Time())
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
            return

        # set current timestamp on goal to avoid extrapolation issues
        self.goal_o.header.stamp = self.get_clock().now().to_msg()

        # compute the coordinates of the goal
        # with respect to the current pose of the robot's base
        # note that the vector [goal_x_b, goal_y_b]^T
        # points from the origin of the robot base frame to the goal
        # and thus per design it has the same direction
        # as the negative gradient of the attractive potential field
        goal_b = tf2_geometry_msgs.do_transform_point(self.goal_o, tf_o_to_b)
        goal_x_b = goal_b.point.x
        goal_y_b = goal_b.point.y

        # compute current Euclidean distance to goal
        # which is simply the magnitude of the 2D vector we just considered
        dist_to_goal = math.sqrt(goal_x_b**2 + goal_y_b**2)
        if do_log:
            self.get_logger().info(f"dist_to_goal={dist_to_goal:.2f}")

        if dist_to_goal < self.approach_threshold:
            # this code block is only relevant when we are near the goal
            # when we are already near the goal we approach it more deliberately
            self.planning_near_goal_counter += 1
            if do_log:
                self.get_logger().info(f"dist_to_goal={dist_to_goal:.2f} < self.approach_threshold={self.approach_threshold:.2f}")

            try:
                # lookup tf from base_link to odom via buffer subscribed to /tf topic
                # this we use to extract current orientation of the robot
                tf_b_to_o = self.tf_buffer.lookup_transform('odom', 'base_link', rclpy.time.Time())
            except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
                return

            # extract current orientation of the robot w.r.t. its initial orientation
            quat = tf_b_to_o.transform.rotation
            _, _, current_theta_o = euler_from_quaternion([quat.x, quat.y, quat.z, quat.w])

            # compare that with the desired orientation
            ang_error = self.normalize_angle(self.goal_theta_o - current_theta_o)

            twist = Twist()
            twist.linear.x = 0.0
            twist.angular.z = self.k_ang * ang_error
            twist.angular.z = max(min(twist.angular.z, self.v_max_angular), -self.v_max_angular)
            self.publisher.publish(twist)

            if abs(ang_error) < self.ang_threshold:
                # distance and orientation are (almost) as requested so the robot reached the goal
                self.get_logger().info(f"abs(ang_error)={abs(ang_error):.2f} < self.ang_threshold={self.ang_threshold:.2f}")
                self.get_logger().info("Goal reached.")
                self.goal_reached = True
            return

        else:
            # this code block is only relevant when we are not yet near the goal
            # compute attractive velocities zeroed at goal
            vx_a_b = self.k_a * (goal_x_b / dist_to_goal) if dist_to_goal > 1e-3 else 0.0
            vy_a_b = self.k_a * (goal_y_b / dist_to_goal) if dist_to_goal > 1e-3 else 0.0

            vx_r_b = 0.0
            vy_r_b = 0.0
            scan_angle_rad = scan.angle_min
            for obstacle_dist in scan.ranges:
                if scan.range_min < obstacle_dist < scan.range_max:
                    if obstacle_dist < self.rho_0:
                        scalar = self.k_r * (1.0 / obstacle_dist - 1.0 / self.rho_0) / (obstacle_dist ** 2)
                        cos_a = math.cos(scan_angle_rad)
                        sin_a = math.sin(scan_angle_rad)
                        vx_r_b += scalar * (-cos_a)
                        vy_r_b += scalar * (-sin_a)
                scan_angle_rad += scan.angle_increment

            v_r = math.sqrt(vx_r_b**2 + vy_r_b**2)
            if v_r > self.v_r_max:
                vx_r_b = (vx_r_b / v_r) * self.v_r_max
                vy_r_b = (vy_r_b / v_r) * self.v_r_max

            vx_total_b = vx_a_b + vx_r_b
            vy_total_b = vy_a_b + vy_r_b

            v_total = math.sqrt(vx_total_b**2 + vy_total_b**2)
            if do_log:
                self.get_logger().info(f"v_total={v_total:.2f}")

            desired_direction = math.atan2(vy_total_b, vx_total_b) if v_total > 0 else 0.0

            twist = Twist()
            twist.linear.x = min(self.v_max_linear, v_total) * math.cos(desired_direction)
            twist.angular.z = self.k_ang * desired_direction
            twist.angular.z = max(min(twist.angular.z, self.v_max_angular), -self.v_max_angular)

            if twist.linear.x < 0:
                twist.linear.x = 0.0

            self.publisher.publish(twist)

    def normalize_angle(self, angle_rad):
        while angle_rad > math.pi:
            angle_rad -= 2 * math.pi
        while angle_rad < -math.pi:
            angle_rad += 2 * math.pi
        return angle_rad

def main(args=None):
    rclpy.init(args=args)
    node = PotentialFieldPlanner()
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    try:
        executor.spin()
    finally:
        executor.shutdown()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
