import numpy as np

import rclpy
from rclpy.executors import MultiThreadedExecutor
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from rclpy.duration import Duration
from geometry_msgs.msg import Twist, PointStamped, TransformStamped
from rclpy.time import Time
from sensor_msgs.msg import LaserScan
import tf2_ros
import tf2_geometry_msgs
from tf_transformations import euler_from_quaternion
import math
import threading
from rcl_interfaces.msg import ParameterDescriptor, SetParametersResult
from rclpy.lifecycle import LifecycleNode, TransitionCallbackReturn

class PotentialFieldPlanner(LifecycleNode):
    def __init__(self):
        super().__init__('potential_field_planner')
        # static goal in odom
        self.goal_o = None
        self.goal_theta_o = None
        self.goal_reached = False
        self.latest_scan = None
        self.lock = threading.Lock()
        self.timer = None
        # subscription to get scan messages
        self.subscription = None
        # publisher to send commands
        self.publisher = None
        # infrastructure to get information on transformations
        self.tf_buffer = None
        self.tf_listener = None
        self.tf_static_buffer = None
        self.tf_static_listener = None
        # counters
        self.planning_counter = 0
        self.planning_near_goal_counter = 0
        self.stuck_counter = 0

    def on_configure(self, state):
        self.get_logger().info('Configuring potential field planner...')
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

        self.latest_scan = None

        # 10hz planning timer
        timer_period = 0.1
        self.timer = self.create_timer(timer_period, self.planning_callback)
        self.get_logger().info('Potential field planner configured successfully.')

        # subscription to get scan messages
        # best effort for consuming sensor data
        qos_sub = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            durability=DurabilityPolicy.VOLATILE
        )
        self.subscription = self.create_subscription(LaserScan, '/scan', self.scan_callback, qos_sub)

        # publisher to send commands
        # reliability for sending messages to cmd_vel
        qos_pub = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            durability=DurabilityPolicy.VOLATILE
        )
        self.publisher = self.create_publisher(Twist, '/cmd_vel', qos_pub)

        # infrastructure to get information on transformations
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.tf_static_buffer = tf2_ros.Buffer()
        self.tf_static_listener = tf2_ros.TransformListener(self.tf_static_buffer, self)

        # counters
        self.planning_counter = 0
        self.planning_near_goal_counter = 0
        self.stuck_counter = 0

        return TransitionCallbackReturn.SUCCESS

    def on_activate(self, state):
        self.get_logger().info('Activating potential field planner...')
        self.timer.start()
        return TransitionCallbackReturn.SUCCESS

    def on_deactivate(self, state):
        self.get_logger().info('Deactivating potential field planner...')
        self.timer.cancel()
        return TransitionCallbackReturn.SUCCESS

    def on_shutdown(self, state):
        self.get_logger().info('Shutting down potential field planner...')
        if self.timer:
            self.timer.cancel()
        return TransitionCallbackReturn.SUCCESS

    def params_callback(self, params):
        for param in params:
            if param.name == 'k_a':
                self.k_a = param.value
            elif param.name == 'rho_0':
                self.rho_0 = param.value
            elif param.name == 'k_r':
                self.k_r = param.value
            elif param.name == 'v_r_max':
                self.v_r_max = param.value
            elif param.name == 'k_ang':
                self.k_ang = param.value
            elif param.name == 'v_max_linear':
                self.v_max_linear = param.value
            elif param.name == 'v_max_angular':
                self.v_max_angular = param.value
            elif param.name == 'approach_threshold':
                self.approach_threshold = param.value
            elif param.name == 'ang_threshold':
                self.ang_threshold = param.value
        return SetParametersResult(successful=True)

    def scan_callback(self, msg):
        # lock for safe scan update
        with self.lock:
            self.latest_scan = msg

    def planning_callback(self):
        # access scan safely
        do_log = self.planning_counter % 10 == 0
        self.planning_counter += 1

        # lock for safe scan access
        with self.lock:
            if self.latest_scan is None:
                return
            # reference to immutable object instead of cloning is okay
            scan = self.latest_scan

        # scan freshness check
        scan_time = scan.header.stamp
        if self.get_clock().now() - Time.from_msg(scan_time) > Duration(seconds=1.0):
            self.get_logger().warning("Stale scan detected! Stopping robot for now to avoid potential accidents.")
            self.publisher.publish(Twist())
            return

        if self.goal_reached:
            if do_log:
                self.get_logger().info("Goal already reached.")
            self.publisher.publish(Twist())
            return

        # check if transform available
        # now = self.get_clock().now()
        scan_time = Time.from_msg(scan.header.stamp)
        if not self.tf_buffer.can_transform('base_link', 'odom', scan_time, timeout=Duration(seconds=1.0)):
            self.get_logger().warning('Currently cannot transform from o to b yet!')
            return

        try:
            # lookup tf from odom to base_link via buffer subscribed to /tf topic
            # this provides transform to map goal from odom to base_link frame
            tf_o_to_b = self.tf_buffer.lookup_transform('base_link', 'odom', scan_time)
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
            if do_log:
                self.get_logger().warning('Failed to lookup tf_o_to_b!')
            return

        try:
            # lookup tf from scanner to base_link via buffer subscribed to /tf_static topic
            # this provides transform to map scans from scanner frame to base_link frame
            tf_s_to_b = self.tf_static_buffer.lookup_transform('base_link', scan.header.frame_id, Time())
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
            if do_log:
                self.get_logger().warning('Failed to lookup tf_s_to_b!')
            return

        self.goal_o.header.stamp = scan.header.stamp

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
                tf_b_to_o = self.tf_buffer.lookup_transform('odom', 'base_link', scan_time)
            except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
                return

            # extract current orientation of the robot w.r.t. its initial orientation
            quat = tf_b_to_o.transform.rotation
            _, _, current_theta_o = euler_from_quaternion([quat.x, quat.y, quat.z, quat.w])

            # compare that with the desired orientation
            ang_error = self.normalize_angle(self.goal_theta_o - current_theta_o)

            twist = Twist()
            twist.linear.x = 0.0
            twist.angular.z = max(min(self.k_ang * ang_error, self.v_max_angular), -self.v_max_angular)
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

            # convert scan to numpy arrays
            ranges = np.array(scan.ranges)
            angles = scan.angle_min + np.arange(len(ranges)) * scan.angle_increment

            # filter valid ranges
            valid_mask = np.isfinite(ranges) & (ranges > scan.range_min) & (ranges < scan.range_max)
            valid_ranges = ranges[valid_mask]
            valid_angles = angles[valid_mask]

            if len(valid_ranges) > 0:
                # compute obstacle positions in scanner frame
                obs_x_s = valid_ranges * np.cos(valid_angles)
                obs_y_s = valid_ranges * np.sin(valid_angles)

                # extract rotation and translation from tf_s_to_b
                rot = tf_s_to_b.transform.rotation
                trans = tf_s_to_b.transform.translation

                # convert quaternion to rotation matrix (2d projection)
                # for 2d case we only need yaw rotation
                _, _, yaw = euler_from_quaternion([rot.x, rot.y, rot.z, rot.w])
                cos_yaw = np.cos(yaw)
                sin_yaw = np.sin(yaw)

                # transform obstacles to base_link frame
                obs_x_b = cos_yaw * obs_x_s - sin_yaw * obs_y_s + trans.x
                obs_y_b = sin_yaw * obs_x_s + cos_yaw * obs_y_s + trans.y

                # compute distances in base_link frame
                obs_dist_b = np.sqrt(obs_x_b ** 2 + obs_y_b ** 2)

                # avoid division by zero
                valid_dist_mask = obs_dist_b > 1e-6
                obs_x_b = obs_x_b[valid_dist_mask]
                obs_y_b = obs_y_b[valid_dist_mask]
                obs_dist_b = obs_dist_b[valid_dist_mask]
                valid_ranges = valid_ranges[valid_dist_mask]

                # filter based on transformed distance
                repulsion_mask = obs_dist_b < self.rho_0
                obs_x_b = obs_x_b[repulsion_mask]
                obs_y_b = obs_y_b[repulsion_mask]
                obs_dist_b_filtered = obs_dist_b[repulsion_mask]

                if len(obs_dist_b_filtered) > 0:
                    # compute repulsive velocity contributions
                    # scalar magnitude based on distance in base_link frame
                    scalars = self.k_r * (1.0 / obs_dist_b_filtered - 1.0 / self.rho_0) / (obs_dist_b_filtered ** 2)

                    # direction away from obstacles in base_link frame (normalized)
                    dir_x_b = -obs_x_b / obs_dist_b_filtered
                    dir_y_b = -obs_y_b / obs_dist_b_filtered

                    # sum all repulsive contributions
                    vx_r_b = np.sum(scalars * dir_x_b)
                    vy_r_b = np.sum(scalars * dir_y_b)
                else:
                    vx_r_b = 0.0
                    vy_r_b = 0.0
            else:
                vx_r_b = 0.0
                vy_r_b = 0.0

            # clamp total repulsive velocity magnitude
            v_r = math.sqrt(vx_r_b ** 2 + vy_r_b ** 2)
            if v_r > self.v_r_max:
                vx_r_b = (vx_r_b / v_r) * self.v_r_max
                vy_r_b = (vy_r_b / v_r) * self.v_r_max

            vx_total_b = vx_a_b + vx_r_b
            vy_total_b = vy_a_b + vy_r_b

            v_total = math.sqrt(vx_total_b**2 + vy_total_b**2)
            if do_log:
                self.get_logger().info(f"v_total={v_total:.2f}")

            # attempt to avoid local minima
            if v_total < 0.005 and dist_to_goal > self.approach_threshold:
                self.stuck_counter += 1
            else:
                self.stuck_counter = 0
            if self.stuck_counter > 50:
                # when stuck rotate in place for a short burst
                self.stuck_counter = 0
                self.get_logger().warning('Trying to escape local minimum!')
                twist = Twist()
                twist.angular.z = 0.7
                self.publisher.publish(twist)
                return

            if v_total < 1e-6:
                desired_direction = 0.0
            else:
                # no need to normalize this angle due to definition of atan2
                desired_direction = math.atan2(vy_total_b, vx_total_b)

            twist = Twist()
            # twist.linear.x = max(0.0, min(self.v_max_linear, v_total * math.cos(desired_direction)))
            twist.linear.x = max(0.0, min(self.v_max_linear, v_total))
            twist.angular.z = max(min(self.k_ang * desired_direction, self.v_max_angular), -self.v_max_angular)

            self.publisher.publish(twist)

    def normalize_angle(self, angle_rad):
        return math.atan2(math.sin(angle_rad), math.cos(angle_rad))
        # while angle_rad > math.pi:
        #     angle_rad -= 2 * math.pi
        # while angle_rad < -math.pi:
        #     angle_rad += 2 * math.pi
        # return angle_rad

def main(args=None):
    print("Begin of main of potential field planner.")
    rclpy.init(args=args)
    node = PotentialFieldPlanner()
    # configure and activate for basic lifecycle flow
    node.configure()
    node.activate()
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    try:
        print("Calling executor spin.")
        executor.spin()
        print("The executor spin returned.")
    finally:
        node.deactivate()
        node.shutdown()
        executor.shutdown()
        node.destroy_node()
        rclpy.shutdown()
        print("End of main of potential field planner.")

if __name__ == '__main__':
    main()
