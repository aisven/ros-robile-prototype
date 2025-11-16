import math
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from rclpy.time import Time
from rcl_interfaces.msg import ParameterDescriptor
from geometry_msgs.msg import Twist, PoseStamped
from sensor_msgs.msg import LaserScan
from tf2_ros import Buffer, TransformListener, TransformException
from tf_transformations import euler_from_quaternion, quaternion_from_euler, quaternion_matrix

# goal pose in odom frame (assignment)
GOAL_X_O = 4.0  # x in odom (m)
GOAL_Y_O = 10.0  # y in odom (m)
GOAL_THETA_O = -1.0  # theta in odom (rad)

class PotentialFieldController(Node):
    def __init__(self):
        super().__init__('potential_field_controller')

        # declare tunable parameters
        self.declare_parameter('k_a', 0.6, ParameterDescriptor(description='attraction gain'))
        self.declare_parameter('k_r', 0.8, ParameterDescriptor(description='repulsion gain'))
        self.declare_parameter('rho_0', 1.5, ParameterDescriptor(description='repulsion threshold (m)'))
        self.declare_parameter('v_r_max', 1.0, ParameterDescriptor(description='max repulsive velocity (m/s)'))
        self.declare_parameter('v_max_linear', 0.6, ParameterDescriptor(description='max linear speed (m/s)'))
        self.declare_parameter('v_max_angular', 1.2, ParameterDescriptor(description='max angular speed (rad/s)'))
        self.declare_parameter('k_ang', 1.2, ParameterDescriptor(description='angular gain when orienting at goal'))
        self.declare_parameter('approach_threshold', 0.25, ParameterDescriptor(description='distance to start orientation control (m)'))
        self.declare_parameter('ang_threshold', 0.08, ParameterDescriptor(description='angular threshold to consider orientation reached (rad)'))
        self.declare_parameter('pos_tolerance', 0.15, ParameterDescriptor(description='position tolerance to consider goal reached (m)'))
        self.declare_parameter('avoid_backwards', True, ParameterDescriptor(description='do not command negative linear.x (avoid moving backwards)'))

        # get parameter values
        self.k_a = self.get_parameter('k_a').value
        self.k_r = self.get_parameter('k_r').value
        self.rho_0 = self.get_parameter('rho_0').value
        self.v_r_max = self.get_parameter('v_r_max').value
        self.v_max_linear = self.get_parameter('v_max_linear').value
        self.v_max_angular = self.get_parameter('v_max_angular').value
        self.k_ang = self.get_parameter('k_ang').value
        self.approach_threshold = self.get_parameter('approach_threshold').value
        self.ang_threshold = self.get_parameter('ang_threshold').value
        self.pos_tolerance = self.get_parameter('pos_tolerance').value
        self.avoid_backwards = self.get_parameter('avoid_backwards').value

        # check use_sim_time
        use_sim_time = self.get_parameter('use_sim_time').value
        self.get_logger().info(f'Using sim time: {use_sim_time}')
        if use_sim_time and self.get_clock().clock_type.name != 'ROS_CLOCK_SIM_TIME':
            self.get_logger().warn('use_sim_time=True but clock not sim-based; check launch')

        # goal pose in odom frame
        quat_o = quaternion_from_euler(0.0, 0.0, GOAL_THETA_O)
        self.goal_pose_o = PoseStamped()
        self.goal_pose_o.header.frame_id = 'odom'
        self.goal_pose_o.pose.position.x = GOAL_X_O
        self.goal_pose_o.pose.position.y = GOAL_Y_O
        self.goal_pose_o.pose.position.z = 0.0
        self.goal_pose_o.pose.orientation.x = quat_o[0]
        self.goal_pose_o.pose.orientation.y = quat_o[1]
        self.goal_pose_o.pose.orientation.z = quat_o[2]
        self.goal_pose_o.pose.orientation.w = quat_o[3]

        # state flags
        self.goal_reached = False

        # tf setup for frame transforms (uses sim time if /use_sim_time param set in launch)
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # publishers and subscribers
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.scan_sub = self.create_subscription(LaserScan, '/scan', self.laser_callback, 10)

        # store latest laser data and frame
        self.laser_data = None
        self.laser_frame = None

        # control timer (10 hz)
        self.control_timer = self.create_timer(0.1, self.control_loop)

        # one-shot timer for initial logging after tf populates
        self.initial_log_timer = self.create_timer(3.0, self.log_initial_info)

    def laser_callback(self, msg):
        # store latest scan data and its frame_id (preserve original timestamp)
        self.laser_data = msg
        self.laser_frame = msg.header.frame_id

    def log_initial_info(self):
        # cancel this timer after first call
        self.initial_log_timer.destroy()

        # get available frames from tf buffer
        try:
            frames_str = self.tf_buffer.all_frames_as_string()
            frames = [f.strip() for f in frames_str.split('\n') if f.strip()]
        except Exception as e:
            frames = [f"error: {e}"]
            self.get_logger().warn(f"tf frames error: {e}")

        # get topic names and types, filter to relevant navigation ones
        try:
            topics = self.get_topic_names_and_types()
            relevant_keys = ['scan', 'odom', 'cmd_vel', 'tf', 'tf_static']
            relevant_topics = [name for name, _ in topics if any(key in name for key in relevant_keys)]
        except Exception as e:
            relevant_topics = [f"error: {e}"]
            self.get_logger().warn(f"topics error: {e}")

        # compile param string
        params_str = (f"k_a={self.k_a}, k_r={self.k_r}, rho_0={self.rho_0}, v_r_max={self.v_r_max}, "
                      f"v_max_linear={self.v_max_linear}, v_max_angular={self.v_max_angular}, "
                      f"k_ang={self.k_ang}, approach_threshold={self.approach_threshold}, "
                      f"ang_threshold={self.ang_threshold}, pos_tolerance={self.pos_tolerance}, "
                      f"avoid_backwards={self.avoid_backwards}")

        # log compiled info at info level
        self.get_logger().info(f"Initialized - Frames: {frames}, Relevant Topics: {relevant_topics}, Params: {params_str}")

    def compute_attractive_b(self, goal_pos_b):
        # compute attractive velocity in base_link frame: k_a * unit vector towards goal from robot (0,0)
        dist = np.linalg.norm(goal_pos_b)
        if dist == 0.0:
            return np.zeros(2)
        # direction scaled by gain (constant speed towards goal)
        return self.k_a * goal_pos_b / dist

    def compute_repulsive_b(self):
        # compute summed repulsive velocities in base_link frame from laser points (batched/vectorized)
        if self.laser_data is None or self.laser_frame is None:
            return np.zeros(2)

        # check if transform is available using scan timestamp
        scan_time = Time.from_msg(self.laser_data.header.stamp)
        timeout = Duration(seconds=0.1)
        if not self.tf_buffer.can_transform('base_link', self.laser_frame, scan_time, timeout):
            self.get_logger().debug("laser to base_link transform not available")
            return np.zeros(2)

        try:
            # lookup transform once for batch
            trans = self.tf_buffer.lookup_transform('base_link', self.laser_frame, scan_time, timeout)
            t = np.array([trans.transform.translation.x, trans.transform.translation.y, trans.transform.translation.z])
            quat = [trans.transform.rotation.x, trans.transform.rotation.y,
                    trans.transform.rotation.z, trans.transform.rotation.w]
            R = quaternion_matrix(quat)[:3, :3]  # rotation matrix (source to target)
        except TransformException as e:
            self.get_logger().debug(f"laser transform failed: {e}")
            return np.zeros(2)

        # prepare angles and ranges as numpy arrays
        n_rays = len(self.laser_data.ranges)
        angles = np.arange(n_rays) * self.laser_data.angle_increment + self.laser_data.angle_min
        ranges_np = np.array(self.laser_data.ranges)

        # mask for valid rays (within rho_0 and above min range)
        valid_mask = (ranges_np < self.rho_0) & (ranges_np > self.laser_data.range_min)
        if not np.any(valid_mask):
            return np.zeros(2)

        # extract valid points in scanner frame
        valid_angles = angles[valid_mask]
        valid_rs = ranges_np[valid_mask]
        point_xs_s = valid_rs * np.cos(valid_angles)
        point_ys_s = valid_rs * np.sin(valid_angles)
        point_zs_s = np.zeros_like(point_xs_s)
        points_s = np.column_stack([point_xs_s, point_ys_s, point_zs_s])

        # batch transform to base_link: p_b = R @ p_s + t (column vectors)
        points_b = (R @ points_s.T).T + t

        # extract 2D obs vectors and distances in base_link
        obs_vecs_b = points_b[:, :2]
        dists = np.linalg.norm(obs_vecs_b, axis=1)

        # re-filter post-transform (in case tf offset makes dist >= rho_0)
        post_mask = dists < self.rho_0
        if not np.any(post_mask):
            return np.zeros(2)

        # compute unit away directions: -obs_vec_b / dist
        unit_away_b = -obs_vecs_b[post_mask] / dists[post_mask, np.newaxis]

        # compute scalars: k_r * (1/dist - 1/rho_0) / dist^2
        post_dists = dists[post_mask]
        scalars = self.k_r * (1.0 / post_dists - 1.0 / self.rho_0) / (post_dists ** 2)

        # individual v_rep_b
        v_reps_b = scalars[:, np.newaxis] * unit_away_b

        # sum all
        rep_total_b = np.sum(v_reps_b, axis=0)

        # cap total repulsive magnitude
        rep_norm = np.linalg.norm(rep_total_b)
        if rep_norm > self.v_r_max:
            rep_total_b = (rep_total_b / rep_norm) * self.v_r_max

        return rep_total_b

    def control_loop(self):
        if self.goal_reached:
            # already reached, keep stopped
            return

        # update goal stamp to current time for latest transform
        self.goal_pose_o.header.stamp = self.get_clock().now().to_msg()
        goal_time = Time.from_msg(self.goal_pose_o.header.stamp)

        # check if odom to base_link transform is available (latest time)
        latest_time = Time()
        timeout = Duration(seconds=0.5)
        if not self.tf_buffer.can_transform('base_link', 'odom', latest_time, timeout):
            self.get_logger().warn("odom to base_link transform not available")
            return

        # lookup current transform odom to base_link for current pose_o
        try:
            t_odom_to_bl = self.tf_buffer.lookup_transform('base_link', 'odom', latest_time, timeout)
            # extract current yaw_o from transform rotation (note: inverted frames, but quat inverse for euler)
            quat_o_inv = [t_odom_to_bl.transform.rotation.x, t_odom_to_bl.transform.rotation.y,
                          t_odom_to_bl.transform.rotation.z, t_odom_to_bl.transform.rotation.w]
            # since lookup is base_link <- odom, rotation is from odom to base_link; euler gives yaw of base w.r.t odom
            _, _, current_yaw_o = euler_from_quaternion(quat_o_inv)
        except TransformException as e:
            self.get_logger().warn(f"odom to base_link transform failed: {e}")
            return

        # check if goal transform is available
        if not self.tf_buffer.can_transform('base_link', 'odom', goal_time, timeout):
            self.get_logger().warn("goal odom to base_link transform not available")
            return

        # transform goal pose to base_link for relative position
        try:
            goal_pose_b = self.tf_buffer.transform(self.goal_pose_o, 'base_link', timeout=Duration(seconds=0.1))
            goal_pos_b_x = goal_pose_b.pose.position.x
            goal_pos_b_y = goal_pose_b.pose.position.y
            goal_pos_b = np.array([goal_pos_b_x, goal_pos_b_y])
            dist = np.linalg.norm(goal_pos_b)
        except TransformException as e:
            self.get_logger().warn(f"goal transform failed: {e}")
            return

        # check if position reached
        if dist < self.pos_tolerance:
            # compute orientation error in odom frame
            delta_yaw_o = GOAL_THETA_O - current_yaw_o
            # normalize to [-pi, pi]
            delta_yaw_o = (delta_yaw_o + math.pi) % (2 * math.pi) - math.pi
            if abs(delta_yaw_o) < self.ang_threshold:
                # fully reached, stop and set flag (log only once)
                if not self.goal_reached:
                    cmd = Twist()
                    self.cmd_pub.publish(cmd)
                    self.goal_reached = True
                    self.get_logger().info('Goal reached and oriented!')
                return
            else:
                # orient in place to absolute goal theta
                cmd = Twist()
                cmd.linear.x = 0.0
                cmd.angular.z = np.clip(self.k_ang * delta_yaw_o, -self.v_max_angular, self.v_max_angular)
                self.cmd_pub.publish(cmd)
                return

        # potential field control (not yet reached position)
        # compute attractive in base_link
        v_attr_b = self.compute_attractive_b(goal_pos_b)
        # compute repulsive in base_link
        v_rep_b = self.compute_repulsive_b()
        # total velocity in base_link
        total_v_b = v_attr_b + v_rep_b
        vx_b, vy_b = total_v_b[0], total_v_b[1]

        # compute steering angle from total velocity
        vel_norm = np.linalg.norm(total_v_b)
        if vel_norm > 0.01:  # avoid div0
            alpha_b = math.atan2(vy_b, vx_b)
        else:
            alpha_b = 0.0

        # compute desired direction to goal position in base_link
        desired_dir_b = math.atan2(goal_pos_b[1], goal_pos_b[0])

        # select angular based on distance: exact to goal dir if approaching, else from total field
        if dist < self.approach_threshold:
            angular_z = np.clip(self.k_ang * desired_dir_b, -self.v_max_angular, self.v_max_angular)
        else:
            angular_z = np.clip(self.k_ang * alpha_b, -self.v_max_angular, self.v_max_angular)

        # linear forward (clip, avoid backwards if set)
        linear_x = vx_b
        if self.avoid_backwards:
            linear_x = max(linear_x, 0.0)
        linear_x = np.clip(linear_x, 0.0 if self.avoid_backwards else -self.v_max_linear, self.v_max_linear)

        # publish twist
        cmd = Twist()
        cmd.linear.x = linear_x
        cmd.angular.z = angular_z
        self.cmd_pub.publish(cmd)

def main(args=None):
    rclpy.init(args=args)
    node = PotentialFieldController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
