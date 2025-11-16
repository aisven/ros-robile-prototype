#!/usr/bin/env python3
import math
import threading
import numpy as np

import rclpy
from rclpy.duration import Duration
from rclpy.time import Time
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from rclpy.lifecycle import LifecycleNode, TransitionCallbackReturn
from rcl_interfaces.msg import ParameterDescriptor, SetParametersResult
from lifecycle_msgs.msg import Transition
from lifecycle_msgs.srv import ChangeState

from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist, PointStamped
import tf2_ros
import tf2_geometry_msgs
from tf_transformations import euler_from_quaternion


class PotentialFieldPlanner(LifecycleNode):
    def __init__(self):
        super().__init__('potential_field_planner')

        # runtime state
        self.latest_scan = None
        self.lock = threading.Lock()
        self.timer = None

        # tf infrastructure
        self.tf_buffer = None
        self.tf_listener = None

        # parameters (defaults)
        self.declare_parameter('k_a', 0.6, ParameterDescriptor(description='attraction gain'))
        self.declare_parameter('k_r', 0.8, ParameterDescriptor(description='repulsion gain'))
        self.declare_parameter('rho_0', 1.5, ParameterDescriptor(description='repulsion threshold (m)'))
        self.declare_parameter('v_r_max', 1.0, ParameterDescriptor(description='max repulsive velocity (m/s)'))
        self.declare_parameter('v_max_linear', 0.6, ParameterDescriptor(description='max linear speed (m/s)'))
        self.declare_parameter('v_max_angular', 1.2, ParameterDescriptor(description='max angular speed (rad/s)'))
        self.declare_parameter('k_ang', 1.2, ParameterDescriptor(description='angular gain when orienting at goal'))
        self.declare_parameter('approach_threshold', 0.25, ParameterDescriptor(description='distance to start orientation control (m)'))
        self.declare_parameter('ang_threshold', 0.08, ParameterDescriptor(description='angular threshold to consider orientation reached (rad)'))
        self.declare_parameter('holonomic', True, ParameterDescriptor(description='whether the robot can move in y (holonomic)'))
        self.declare_parameter('avoid_backwards', True, ParameterDescriptor(description='do not command negative linear.x (avoid moving backwards)'))

        # read params
        self._read_params()
        self.add_on_set_parameters_callback(self.params_callback)

        # goal in odom frame (assignment)
        self.goal_o = PointStamped()
        self.goal_o.header.frame_id = 'odom'
        self.goal_o.point.x = 4.0
        self.goal_o.point.y = 10.0
        self.goal_o.point.z = 0.0
        self.goal_theta_o = -1.0

        self.goal_reached = False

        # io placeholders (set up on configure)
        self.scan_sub = None
        self.cmd_pub = None

        # counters for logging & escape logic
        self.planning_counter = 0
        self.stuck_counter = 0
        self.planning_near_goal_counter = 0

    def _read_params(self):
        self.k_a = self.get_parameter('k_a').value
        self.k_r = self.get_parameter('k_r').value
        self.rho_0 = self.get_parameter('rho_0').value
        self.v_r_max = self.get_parameter('v_r_max').value
        self.v_max_linear = self.get_parameter('v_max_linear').value
        self.v_max_angular = self.get_parameter('v_max_angular').value
        self.k_ang = self.get_parameter('k_ang').value
        self.approach_threshold = self.get_parameter('approach_threshold').value
        self.ang_threshold = self.get_parameter('ang_threshold').value
        self.holonomic = self.get_parameter('holonomic').value
        self.avoid_backwards = self.get_parameter('avoid_backwards').value

    def params_callback(self, params):
        for p in params:
            if p.name == 'k_a':
                self.k_a = p.value
            elif p.name == 'k_r':
                self.k_r = p.value
            elif p.name == 'rho_0':
                self.rho_0 = p.value
            elif p.name == 'v_r_max':
                self.v_r_max = p.value
            elif p.name == 'v_max_linear':
                self.v_max_linear = p.value
            elif p.name == 'v_max_angular':
                self.v_max_angular = p.value
            elif p.name == 'k_ang':
                self.k_ang = p.value
            elif p.name == 'approach_threshold':
                self.approach_threshold = p.value
            elif p.name == 'ang_threshold':
                self.ang_threshold = p.value
            elif p.name == 'holonomic':
                self.holonomic = p.value
            elif p.name == 'avoid_backwards':
                self.avoid_backwards = p.value
        return SetParametersResult(successful=True)

    def on_configure(self, state):
        self.get_logger().info('configuring potential field planner...')
        # qos for scan (best-effort typical for lidar)
        qos_sub = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            durability=DurabilityPolicy.VOLATILE
        )
        self.scan_sub = self.create_subscription(LaserScan, '/scan', self.scan_callback, qos_sub)

        # cmd_vel publisher
        qos_pub = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            durability=DurabilityPolicy.VOLATILE
        )
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', qos_pub)

        # single tf buffer/listener (do not rely solely on spin_thread on all systems)
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # 10 hz planning timer
        self.timer = self.create_timer(0.1, self.planning_callback)

        self.get_logger().info('potential field planner configured successfully.')
        return TransitionCallbackReturn.SUCCESS

    def on_activate(self, state):
        self.get_logger().info('activating potential field planner...')
        try:
            return TransitionCallbackReturn.SUCCESS
        except Exception as e:
            self.get_logger().error(f'error during activation: {e}')
            return TransitionCallbackReturn.FAILURE

    def on_deactivate(self, state):
        self.get_logger().info('deactivating potential field planner...')
        try:
            if self.cmd_pub is not None:
                self.cmd_pub.publish(Twist())
            return TransitionCallbackReturn.SUCCESS
        except Exception as e:
            self.get_logger().error(f'error during deactivation: {e}')
            return TransitionCallbackReturn.FAILURE

    def on_cleanup(self, state):
        self.get_logger().info('cleaning up potential field planner...')
        try:
            if self.timer is not None:
                self.timer.cancel()
                self.timer = None
            if self.scan_sub is not None:
                self.scan_sub.destroy()
                self.scan_sub = None
            if self.cmd_pub is not None:
                self.cmd_pub.destroy()
                self.cmd_pub = None
            if self.tf_buffer is not None:
                try:
                    self.tf_buffer.clear()
                except Exception:
                    pass
                self.tf_buffer = None
            self.latest_scan = None
            return TransitionCallbackReturn.SUCCESS
        except Exception as e:
            self.get_logger().error(f'error during cleanup: {e}')
            return TransitionCallbackReturn.FAILURE

    def on_shutdown(self, state):
        self.get_logger().info('shutting down potential field planner...')
        return self.on_cleanup(state)

    def scan_callback(self, msg: LaserScan):
        # store latest scan under lock
        with self.lock:
            self.latest_scan = msg

    def planning_callback(self):
        do_log = (self.planning_counter % 10 == 0)
        self.planning_counter += 1

        # copy scan under lock
        with self.lock:
            scan = self.latest_scan
        if scan is None:
            if do_log:
                self.get_logger().debug('no scan data yet')
            return

        # handle stamp robustness: use header if present, otherwise receipt time
        receipt_time = self.get_clock().now()
        if scan.header.stamp.sec == 0 and scan.header.stamp.nanosec == 0:
            if do_log:
                self.get_logger().info('scan header stamp is zero; using receipt time')
            scan_time = receipt_time
        else:
            scan_time = Time.from_msg(scan.header.stamp)

        # age check
        now = self.get_clock().now()
        age = (now - scan_time).nanoseconds / 1e9
        if do_log:
            scan_sec = scan_time.nanoseconds // int(1e9)
            scan_ms = (scan_time.nanoseconds % int(1e9)) // int(1e6)
            now_sec = now.nanoseconds // int(1e9)
            now_ms = (now.nanoseconds % int(1e9)) // int(1e6)
            self.get_logger().info(f'scan stamp: {scan_sec}.{scan_ms:03d}, now: {now_sec}.{now_ms:03d}, age: {age:.3f}s')

        if now - scan_time > Duration(seconds=1.0):
            if do_log:
                self.get_logger().warning('stale scan detected; publishing zero velocity for safety')
            self.cmd_pub.publish(Twist())
            return

        if self.goal_reached:
            if do_log:
                self.get_logger().info('goal already reached')
            self.cmd_pub.publish(Twist())
            return

        # request latest odom->base_link transform; wait briefly if necessary
        try:
            if not self.tf_buffer.can_transform('base_link', 'odom', Time(), timeout=Duration(seconds=0.2)):
                if do_log:
                    self.get_logger().warning('odom->base_link transform not available yet')
                return
            tf_o_to_b = self.tf_buffer.lookup_transform('base_link', 'odom', Time())
        except Exception as e:
            if do_log:
                self.get_logger().warning(f'failed to obtain odom->base_link transform: {e}')
            return

        # request latest scanner_frame->base_link transform
        try:
            if not self.tf_buffer.can_transform('base_link', scan.header.frame_id, Time(), timeout=Duration(seconds=0.2)):
                if do_log:
                    self.get_logger().warning('scanner->base_link transform not available yet')
                return
            tf_s_to_b = self.tf_buffer.lookup_transform('base_link', scan.header.frame_id, Time())
        except Exception as e:
            if do_log:
                self.get_logger().warning(f'failed to obtain scanner->base_link transform: {e}')
            return

        # transform goal into base_link frame so all vectors are in base_link
        # set goal header stamp to scan_time (not strictly necessary for latest, but keeps semantics)
        goal_o_copy = PointStamped()
        goal_o_copy.header.frame_id = self.goal_o.header.frame_id
        goal_o_copy.header.stamp = scan.header.stamp
        goal_o_copy.point.x = self.goal_o.point.x
        goal_o_copy.point.y = self.goal_o.point.y
        goal_o_copy.point.z = self.goal_o.point.z
        try:
            goal_b = tf2_geometry_msgs.do_transform_point(goal_o_copy, tf_o_to_b)
        except Exception as e:
            if do_log:
                self.get_logger().warning(f'failed to transform goal into base_link: {e}')
            return

        goal_x_b = goal_b.point.x
        goal_y_b = goal_b.point.y
        dist_to_goal = math.hypot(goal_x_b, goal_y_b)
        if do_log:
            self.get_logger().info(f'distance to goal (base_link) = {dist_to_goal:.3f} m')

        # near-goal orientation control
        if dist_to_goal < self.approach_threshold:
            self.planning_near_goal_counter += 1
            # get current orientation: lookup odom <- base_link transform
            try:
                if not self.tf_buffer.can_transform('odom', 'base_link', Time(), timeout=Duration(seconds=0.2)):
                    return
                tf_b_to_o = self.tf_buffer.lookup_transform('odom', 'base_link', Time())
            except Exception:
                return
            quat = tf_b_to_o.transform.rotation
            _, _, current_theta_o = euler_from_quaternion([quat.x, quat.y, quat.z, quat.w])
            ang_err = self.normalize_angle(self.goal_theta_o - current_theta_o)

            twist = Twist()
            twist.linear.x = 0.0
            twist.linear.y = 0.0
            twist.angular.z = max(min(self.k_ang * ang_err, self.v_max_angular), -self.v_max_angular)
            self.cmd_pub.publish(twist)

            if abs(ang_err) < self.ang_threshold:
                self.get_logger().info('goal reached (position & orientation within thresholds)')
                self.goal_reached = True
            return

        # attractive field in base_link frame (direction toward goal, speed scaled smoothly with distance)
        if dist_to_goal > 1e-6:
            ux = goal_x_b / dist_to_goal
            uy = goal_y_b / dist_to_goal
            v_attr_mag = min(self.k_a * dist_to_goal, self.v_max_linear)
            vx_a_b = v_attr_mag * ux
            vy_a_b = v_attr_mag * uy
        else:
            vx_a_b = 0.0
            vy_a_b = 0.0

        # process scan into repulsive contributions (all in base_link)
        ranges = np.array(scan.ranges, dtype=np.float64)
        angles = scan.angle_min + np.arange(len(ranges)) * scan.angle_increment

        # filter valid ranges
        finite_mask = np.isfinite(ranges)
        in_range_mask = (ranges > scan.range_min) & (ranges < scan.range_max)
        valid_mask = finite_mask & in_range_mask

        if not np.any(valid_mask):
            vx_r_b = 0.0
            vy_r_b = 0.0
        else:
            valid_ranges = ranges[valid_mask]
            valid_angles = angles[valid_mask]

            obs_x_s = valid_ranges * np.cos(valid_angles)
            obs_y_s = valid_ranges * np.sin(valid_angles)

            rot = tf_s_to_b.transform.rotation
            trans = tf_s_to_b.transform.translation
            _, _, yaw = euler_from_quaternion([rot.x, rot.y, rot.z, rot.w])
            cos_y = math.cos(yaw)
            sin_y = math.sin(yaw)
            obs_x_b = cos_y * obs_x_s - sin_y * obs_y_s + trans.x
            obs_y_b = sin_y * obs_x_s + cos_y * obs_y_s + trans.y

            obs_dist_b = np.hypot(obs_x_b, obs_y_b)
            nonzero_mask = obs_dist_b > 1e-6
            obs_x_b = obs_x_b[nonzero_mask]
            obs_y_b = obs_y_b[nonzero_mask]
            obs_dist_b = obs_dist_b[nonzero_mask]

            rep_mask = obs_dist_b < self.rho_0
            if np.any(rep_mask):
                obs_x_b = obs_x_b[rep_mask]
                obs_y_b = obs_y_b[rep_mask]
                obs_dist_b = obs_dist_b[rep_mask]

                scalars = self.k_r * (1.0 / obs_dist_b - 1.0 / self.rho_0) / (obs_dist_b ** 2)
                dir_x = obs_x_b / obs_dist_b
                dir_y = obs_y_b / obs_dist_b
                away_x = -dir_x
                away_y = -dir_y

                vx_r_b = float(np.sum(scalars * away_x))
                vy_r_b = float(np.sum(scalars * away_y))
            else:
                vx_r_b = 0.0
                vy_r_b = 0.0

        # clamp repulsive magnitude
        v_r = math.hypot(vx_r_b, vy_r_b)
        if v_r > self.v_r_max and v_r > 0.0:
            vx_r_b = (vx_r_b / v_r) * self.v_r_max
            vy_r_b = (vy_r_b / v_r) * self.v_r_max

        # total desired velocity in base_link frame
        vx_total_b = vx_a_b + vx_r_b
        vy_total_b = vy_a_b + vy_r_b
        v_total = math.hypot(vx_total_b, vy_total_b)
        if do_log:
            self.get_logger().info(f'v_total={v_total:.3f} vx_a={vx_a_b:.3f} vy_a={vy_a_b:.3f} vx_r={vx_r_b:.3f} vy_r={vy_r_b:.3f}')

        # escape local minima: attempt simple wiggle/rotation
        if v_total < 0.005 and dist_to_goal > self.approach_threshold:
            self.stuck_counter += 1
        else:
            self.stuck_counter = 0
        if self.stuck_counter > 40:
            self.stuck_counter = 0
            self.get_logger().warning('trying to escape local minimum')
            twist = Twist()
            if self.holonomic:
                twist.linear.x = 0.0
                twist.linear.y = 0.1
            else:
                twist.angular.z = 0.7
            self.cmd_pub.publish(twist)
            return

        # desired direction in base_link frame
        if v_total < 1e-9:
            desired_dir = 0.0
        else:
            desired_dir = math.atan2(vy_total_b, vx_total_b)

        # build command depending on holonomic capability
        twist = Twist()
        if self.holonomic:
            vx_cmd = max(-self.v_max_linear, min(self.v_max_linear, vx_total_b))
            vy_cmd = max(-self.v_max_linear, min(self.v_max_linear, vy_total_b))
            if self.avoid_backwards and vx_cmd < 0.0:
                vx_cmd = 0.0
            twist.linear.x = vx_cmd
            twist.linear.y = vy_cmd
            twist.angular.z = 0.0
        else:
            v_forward = vx_total_b
            if self.avoid_backwards:
                v_forward = max(0.0, v_forward)
            v_forward = max(-self.v_max_linear, min(self.v_max_linear, v_forward))
            w_cmd = max(min(self.k_ang * desired_dir, self.v_max_angular), -self.v_max_angular)
            twist.linear.x = v_forward
            twist.linear.y = 0.0
            twist.angular.z = w_cmd

        # near-goal safety stop (position)
        if dist_to_goal < self.approach_threshold:
            twist.linear.x = 0.0
            twist.linear.y = 0.0
            twist.angular.z = 0.0

        self.cmd_pub.publish(twist)

    @staticmethod
    def normalize_angle(angle_rad):
        return math.atan2(math.sin(angle_rad), math.cos(angle_rad))


def main(args=None):
    rclpy.init(args=args)

    # create lifecycle node
    node = PotentialFieldPlanner()

    # create separate small client node to call lifecycle transition services
    client_node = rclpy.create_node('lifecycle_client')
    change_state_client = client_node.create_client(ChangeState, f'/{node.get_name()}/change_state')

    if not change_state_client.wait_for_service(timeout_sec=5.0):
        client_node.get_logger().error('timeout. change_state service not available!')
        # fallback: run node without lifecycle transitions (spin it directly)
        try:
            rclpy.spin(node)
        finally:
            node.destroy_node()
            client_node.destroy_node()
            rclpy.shutdown()
        return

    def call_transition(transition_id: int):
        req = ChangeState.Request()
        req.transition.id = transition_id
        future = change_state_client.call_async(req)
        rclpy.spin_until_future_complete(client_node, future, timeout_sec=5.0)
        if future.result() is not None:
            return future.result().success
        return False

    # configure
    if call_transition(Transition.TRANSITION_CONFIGURE):
        client_node.get_logger().info('node configured')
    else:
        client_node.get_logger().error('node failed to configure')
        node.destroy_node()
        client_node.destroy_node()
        rclpy.shutdown()
        return

    # activate
    if call_transition(Transition.TRANSITION_ACTIVATE):
        client_node.get_logger().info('node activated')
    else:
        client_node.get_logger().error('node failed to activate')
        node.destroy_node()
        client_node.destroy_node()
        rclpy.shutdown()
        return

    try:
        rclpy.spin(node)
    finally:
        # deactivate -> cleanup
        call_transition(Transition.TRANSITION_DEACTIVATE)
        call_transition(Transition.TRANSITION_CLEANUP)
        node.destroy_node()
        client_node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
