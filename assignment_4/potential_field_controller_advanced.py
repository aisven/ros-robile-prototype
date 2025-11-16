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

# code format follows black default with max line length 160
# black --line-length 160

# goal pose in odom frame (assignment)
GOAL_X_O = 4.0  # x in odom (m)
GOAL_Y_O = 10.0  # y in odom (m)
GOAL_THETA_O = -1.0  # theta in odom (rad)

# postfix _o indicates odom frame
# postfix _b indicates base frame aka. base_link
# postfix _s indicates scanner frame aka. laser frame etc.


class PotentialFieldController(Node):
    def __init__(self):
        super().__init__("potential_field_controller")

        # declare node parameters
        self.declare_parameter("k_a", 0.6, ParameterDescriptor(description="attraction gain"))
        self.declare_parameter("k_r", 0.8, ParameterDescriptor(description="repulsion gain"))
        self.declare_parameter("rho_0", 1.5, ParameterDescriptor(description="repulsion threshold (m)"))
        self.declare_parameter("v_r_max", 1.0, ParameterDescriptor(description="max repulsive velocity (m/s)"))
        self.declare_parameter("v_max_linear", 0.6, ParameterDescriptor(description="max linear speed (m/s)"))
        self.declare_parameter("v_max_angular", 1.2, ParameterDescriptor(description="max angular speed (rad/s)"))
        self.declare_parameter("k_ang", 1.2, ParameterDescriptor(description="angular gain when orienting at goal"))
        self.declare_parameter("approach_threshold", 0.25, ParameterDescriptor(description="distance to start orientation control (m)"))
        self.declare_parameter("ang_threshold", 0.08, ParameterDescriptor(description="angular threshold to consider orientation reached (rad)"))
        self.declare_parameter("pos_tolerance", 0.15, ParameterDescriptor(description="position tolerance to consider goal reached (m)"))
        self.declare_parameter("avoid_backwards", True, ParameterDescriptor(description="do not command negative linear.x (avoid moving backwards)"))

        # get parameter values
        self.k_a = self.get_parameter("k_a").value
        self.k_r = self.get_parameter("k_r").value
        self.rho_0 = self.get_parameter("rho_0").value
        self.v_r_max = self.get_parameter("v_r_max").value
        self.v_max_linear = self.get_parameter("v_max_linear").value
        self.v_max_angular = self.get_parameter("v_max_angular").value
        self.k_ang = self.get_parameter("k_ang").value
        self.approach_threshold = self.get_parameter("approach_threshold").value
        self.ang_threshold = self.get_parameter("ang_threshold").value
        self.pos_tolerance = self.get_parameter("pos_tolerance").value
        self.avoid_backwards = self.get_parameter("avoid_backwards").value

        # goal pose in odom frame
        quat_o = quaternion_from_euler(0.0, 0.0, GOAL_THETA_O)
        self.goal_pose_o = PoseStamped()
        self.goal_pose_o.header.frame_id = "odom"
        self.goal_pose_o.header.stamp = self.get_clock().now().to_msg()
        self.goal_pose_o.pose.position.x = GOAL_X_O
        self.goal_pose_o.pose.position.y = GOAL_Y_O
        self.goal_pose_o.pose.position.z = 0.0
        self.goal_pose_o.pose.orientation.x = quat_o[0]
        self.goal_pose_o.pose.orientation.y = quat_o[1]
        self.goal_pose_o.pose.orientation.z = quat_o[2]
        self.goal_pose_o.pose.orientation.w = quat_o[3]

        # flag
        self.goal_reached = False

        # infrastructure for getting transformation information
        # uses sim time if /use_sim_time param set in launch file
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # infrastructure to publish velocity commands to the robot
        self.v_command_publisher = self.create_publisher(Twist, "/cmd_vel", 10)

        # infrastructure for getting LiDAR scan data
        self.scan_subscription = self.create_subscription(LaserScan, "/scan", self.laser_callback, 10)
        self.laser_data = None
        self.laser_frame = None

        # control timer (2 hz)
        self.control_timer = self.create_timer(0.5, self.control_loop)

        # one-shot timer for initial logging after tf populates
        self.initial_log_statement_written = False
        self.initial_log_statement_timer = self.create_timer(5.0, self.log_initial_info)

    def laser_callback(self, msg):
        # store latest scan data and its frame_id (preserve original timestamp)
        self.laser_data = msg
        self.laser_frame = msg.header.frame_id

    def log_initial_info(self):
        # only log once
        if self.initial_log_statement_written:
            self.initial_log_statement_timer.destroy()
            return
        self.initial_log_statement_written = True

        # cancel this timer after first call
        self.initial_log_statement_timer.destroy()

        # get available frames from tf buffer
        try:
            all_frames = self.tf_buffer.all_frames_as_string()
            relevant_frames = [f.strip() for f in all_frames.split("\n") if f.strip()]
        except Exception as e:
            relevant_frames = ["error"]
            self.get_logger().warn(f"Ignoring error during checking frames. {e}")

        # get topic names and types, filter to relevant navigation ones
        try:
            all_topics = self.get_topic_names_and_types()
            relevant_keys = ["scan", "odom", "cmd_vel", "tf", "tf_static"]
            relevant_topics = [name for name, _ in all_topics if any(key in name for key in relevant_keys)]
        except Exception as e:
            relevant_topics = ["error"]
            self.get_logger().warn(f"Ignoring error during checking topics. {e}")

        # compile param string
        use_sim_time = self.get_parameter("use_sim_time").value

        params_str = (
            f"k_a={self.k_a}, k_r={self.k_r}, rho_0={self.rho_0}, v_r_max={self.v_r_max}, "
            f"v_max_linear={self.v_max_linear}, v_max_angular={self.v_max_angular}, "
            f"k_ang={self.k_ang}, approach_threshold={self.approach_threshold}, "
            f"ang_threshold={self.ang_threshold}, pos_tolerance={self.pos_tolerance}, "
            f"avoid_backwards={self.avoid_backwards}, use_sim_time={use_sim_time}"
        )

        # check use_sim_time
        clock_type_name = self.get_clock().clock_type.name
        if use_sim_time and clock_type_name != "ROS_CLOCK_SIM_TIME":
            self.get_logger().warn(f"use_sim_time=True but clock not sim-based! clock_type_name={clock_type_name}.")

        # log compiled info at info level
        self.get_logger().info(f"Node initialized.\nFrames: {relevant_frames}\nTopics: {relevant_topics}\nParameters: {params_str}")

    def compute_attractive_b(self, goal_pos_b):
        # compute attractive velocity in base frame
        # goal_pos_b is the 2D position of the goal in the base frame

        # current distance to goal
        distance_robot_to_goal_b = np.linalg.norm(goal_pos_b)
        if distance_robot_to_goal_b == 0.0:
            return np.zeros(2)

        # unit vector
        direction_robot_to_goal = goal_pos_b / distance_robot_to_goal_b

        # direction scaled by gain (constant speed towards goal)
        return self.k_a * direction_robot_to_goal

    def compute_repulsive_b(self):
        # compute summed repulsive velocities in base_link frame from laser points (batched/vectorized)
        if self.laser_data is None or self.laser_frame is None:
            self.get_logger().warning("No scan data. Returning zero repulsive force.")
            return np.zeros(2)

        # check if transform from scanner to base is available
        scan_time = Time.from_msg(self.laser_data.header.stamp)
        if not self.tf_buffer.can_transform("base_link", self.laser_frame, scan_time, Duration(seconds=0.1)):
            self.get_logger().warning("tf_s_to_b unavailable")
            return np.zeros(2)

        # lookup transform from scanner to base
        try:
            tf_s_to_b = self.tf_buffer.lookup_transform("base_link", self.laser_frame, scan_time, Duration(seconds=0.1))
            t = np.array([tf_s_to_b.transform.translation.x, tf_s_to_b.transform.translation.y, tf_s_to_b.transform.translation.z])
            quat = [tf_s_to_b.transform.rotation.x, tf_s_to_b.transform.rotation.y, tf_s_to_b.transform.rotation.z, tf_s_to_b.transform.rotation.w]
            R = quaternion_matrix(quat)[:3, :3]  # rotation matrix (source to target)
        except TransformException as e:
            self.get_logger().warning(f"Failed to get rotation matrix from laser frame to base frame. {e}")
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
        scalars = self.k_r * (1.0 / post_dists - 1.0 / self.rho_0) / (post_dists**2)

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

        # check if transform from base to odom is available
        latest_time = Time()
        if not self.tf_buffer.can_transform("odom", "base_link", latest_time, Duration(seconds=0.1)):
            self.get_logger().warning("tf_b_to_o unavailable.")
            return

        # extract current_yaw_o from tf_b_to_o
        try:
            tf_b_to_o = self.tf_buffer.lookup_transform("odom", "base_link", latest_time, Duration(seconds=0.1))
            # extract current yaw_o from transform rotation
            quat_o = [tf_b_to_o.transform.rotation.x, tf_b_to_o.transform.rotation.y, tf_b_to_o.transform.rotation.z, tf_b_to_o.transform.rotation.w]
            # rotation from odom to base_link; euler gives yaw of base w.r.t odom
            _, _, current_yaw_o = euler_from_quaternion(quat_o)
        except TransformException as e:
            self.get_logger().warning(f"Failed to extract current_yaw_o from tf_b_to_o. {e}")
            return

        # check if transform from odom to base is available
        if not self.tf_buffer.can_transform("base_link", "odom", goal_time, Duration(seconds=0.1)):
            self.get_logger().warning("tf_o_to_b unavailable.")
            return

        # transform goal pose to base frame
        try:
            goal_pose_b = self.tf_buffer.transform(self.goal_pose_o, "base_link", Duration(seconds=0.1))
            goal_position_b_x = goal_pose_b.pose.position.x
            goal_position_b_y = goal_pose_b.pose.position.y
            goal_position_b = np.array([goal_position_b_x, goal_position_b_y])
            distance_robot_to_goal_b = np.linalg.norm(goal_position_b)
        except TransformException as e:
            self.get_logger().warning(f"Failed to transform goal pose to base frame with tf_o_to_b. {e}")
            return

        # check if position reached
        if distance_robot_to_goal_b < self.pos_tolerance:
            # compute orientation error in odom frame
            delta_yaw_o = GOAL_THETA_O - current_yaw_o
            # normalize to [-pi, pi]
            delta_yaw_o = (delta_yaw_o + math.pi) % (2 * math.pi) - math.pi
            if abs(delta_yaw_o) < self.ang_threshold:
                # fully reached, stop and set flag (log only once)
                if not self.goal_reached:
                    v_command = Twist()
                    self.v_command_publisher.publish(v_command)
                    self.goal_reached = True
                    self.get_logger().info("Goal reached and oriented!")
                return
            else:
                # orient in place to absolute goal theta
                v_command = Twist()
                v_command.linear.x = 0.0
                v_command.angular.z = np.clip(self.k_ang * delta_yaw_o, -self.v_max_angular, self.v_max_angular)
                self.v_command_publisher.publish(v_command)
                return

        # potential field control (not yet reached position)
        # compute attractive force in base frame
        v_a_b = self.compute_attractive_b(goal_position_b)
        # compute repulsive force in base frame
        v_r_b = self.compute_repulsive_b()
        # total velocity in base frame
        v_total_b = v_a_b + v_r_b
        v_total_x_b, v_total_y_b = v_total_b[0], v_total_b[1]

        # compute steering angle from total velocity
        v_total_magnitude = np.linalg.norm(v_total_b)
        # avoid division by zero or very small values
        if v_total_magnitude > 0.005:
            alpha_b = math.atan2(v_total_y_b, v_total_x_b)
        else:
            alpha_b = 0.0

        # compute desired direction to goal position in base frame
        desired_direction_angle_b = math.atan2(goal_position_b[1], goal_position_b[0])

        # select angular based on distance: exact to goal dir if approaching, else from total field
        if distance_robot_to_goal_b < self.approach_threshold:
            v_angular_z_b = np.clip(self.k_ang * desired_direction_angle_b, -self.v_max_angular, self.v_max_angular)
        else:
            v_angular_z_b = np.clip(self.k_ang * alpha_b, -self.v_max_angular, self.v_max_angular)

        # linear forward (clip, avoid backwards if set)
        v_linear_x_b = v_total_x_b
        if self.avoid_backwards:
            v_linear_x_b = max(v_linear_x_b, 0.0)
        v_linear_x_b = np.clip(v_linear_x_b, 0.0 if self.avoid_backwards else -self.v_max_linear, self.v_max_linear)

        # publish twist
        v_command = Twist()
        v_command.linear.x = v_linear_x_b
        v_command.angular.z = v_angular_z_b
        self.v_command_publisher.publish(v_command)


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


if __name__ == "__main__":
    main()
