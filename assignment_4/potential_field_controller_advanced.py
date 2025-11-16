import math
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from rcl_interfaces.msg import ParameterDescriptor
from geometry_msgs.msg import Twist, PoseStamped, PointStamped
from sensor_msgs.msg import LaserScan
from tf2_ros import Buffer, TransformListener, TransformException
from tf2_ros.buffer import Time
from tf_transformations import euler_from_quaternion, quaternion_from_euler

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

        # goal pose in odom frame
        quat_o = quaternion_from_euler(0.0, 0.0, GOAL_THETA_O)
        self.goal_pose_o = PoseStamped()
        self.goal_pose_o.header.frame_id = 'odom'
        self.goal_pose_o.header.stamp = self.get_clock().now().to_msg()
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
        # store latest scan data and its frame_id
        self.laser_data = msg
        self.laser_frame = msg.header.frame_id

    def log_initial_info(self):
        # cancel this timer after first call
        self.initial_log_timer.cancel()

        # get available frames from tf buffer
        try:
            frames_str = self.tf_buffer.all_frames_as_string()
            frames = [f.strip() for f in frames_str.split('\n') if f.strip()]
        except Exception as e:
            frames = [f"error: {e}"]
            self.get_logger().warn(f"tf frames error: {e}")

        # get all topic names
        topics = self.get_topic_names_and_namespaces()
        topic_names = [t[0] for t in topics]

        # compile param string
        params_str = (f"k_a={self.k_a}, k_r={self.k_r}, rho_0={self.rho_0}, v_r_max={self.v_r_max}, "
                      f"v_max_linear={self.v_max_linear}, v_max_angular={self.v_max_angular}, "
                      f"k_ang={self.k_ang}, approach_threshold={self.approach_threshold}, "
                      f"ang_threshold={self.ang_threshold}, pos_tolerance={self.pos_tolerance}, "
                      f"avoid_backwards={self.avoid_backwards}")

        # log compiled info at info level
        self.get_logger().info(f"Initialized - Frames: {frames}, Topics: {topic_names}, Params: {params_str}")

    def compute_attractive_b(self, goal_pos_b):
        # compute attractive velocity in base_link frame: k_a * unit vector towards goal from robot (0,0)
        dx_b, dy_b = goal_pos_b[0], goal_pos_b[1]
        dist = math.hypot(dx_b, dy_b)
        if dist == 0.0:
            return np.zeros(2)
        # direction scaled by gain (note: assignment has -k_a * unit_(q - q_goal), but since q_goal - q = goal_pos_b, unit_(goal_pos_b))
        return self.k_a * np.array([dx_b, dy_b]) / dist

    def compute_repulsive_b(self):
        # compute summed repulsive velocities in base_link frame from laser points
        if self.laser_data is None or self.laser_frame is None:
            return np.zeros(2)
        rep_total_b = np.zeros(2)
        ranges = self.laser_data.ranges
        angle_min = self.laser_data.angle_min
        angle_inc = self.laser_data.angle_increment
        # set stamp for transforms (use latest)
        self.laser_data.header.stamp = self.get_clock().now().to_msg()
        for i, r in enumerate(ranges):
            if r < self.rho_0 and r > self.laser_data.range_min:
                # compute angle for this ray
                theta = angle_min + i * angle_inc
                # create point in scanner frame
                point_s = PointStamped()
                point_s.header.frame_id = self.laser_frame
                point_s.header.stamp = self.laser_data.header.stamp
                point_s.point.x = r * math.cos(theta)
                point_s.point.y = r * math.sin(theta)
                point_s.point.z = 0.0
                try:
                    # transform obstacle point to base_link frame
                    point_b = self.tf_buffer.transform(point_s, 'base_link', timeout=Duration(seconds=0.1))
                    obs_x_b = point_b.point.x
                    obs_y_b = point_b.point.y
                    obs_vec_b = np.array([obs_x_b, obs_y_b])
                    dist_obs = np.linalg.norm(obs_vec_b)
                    if dist_obs == 0.0 or dist_obs >= self.rho_0:
                        continue
                    # direction: unit_(q - q_o) = - unit_obs_vec_b (away from obstacle)
                    unit_away_b = -obs_vec_b / dist_obs
                    # scalar per formula: k_r * (1/dist - 1/rho_0) * 1/dist^2
                    scalar = self.k_r * (1.0 / dist_obs - 1.0 / self.rho_0) / (dist_obs ** 2)
                    v_rep_b = scalar * unit_away_b
                    # cap individual repulsive magnitude
                    rep_norm = np.linalg.norm(v_rep_b)
                    if rep_norm > self.v_r_max:
                        v_rep_b = (v_rep_b / rep_norm) * self.v_r_max
                    # accumulate
                    rep_total_b += v_rep_b
                except TransformException as e:
                    # skip if transform fails (e.g., frames not ready)
                    self.get_logger().debug(f"transform failed for ray {i}: {e}")
                    continue
        return rep_total_b

    def control_loop(self):
        if self.goal_reached:
            # already reached, keep stopped
            return

        # lookup current transform odom to base_link for current pose_o
        try:
            # use current time with short timeout
            t_odom_to_bl = self.tf_buffer.lookup_transform('odom', 'base_link', Time(0, 0), timeout=Duration(seconds=0.5))
            # extract current yaw_o from transform rotation
            quat_o = [t_odom_to_bl.transform.rotation.x, t_odom_to_bl.transform.rotation.y,
                      t_odom_to_bl.transform.rotation.z, t_odom_to_bl.transform.rotation.w]
            _, _, current_yaw_o = euler_from_quaternion(quat_o)
        except TransformException as e:
            self.get_logger().warn(f"odom to base_link transform failed: {e}")
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
                # fully reached, stop and set flag
                cmd = Twist()
                self.cmd_pub.publish(cmd)
                self.goal_reached = True
                self.get_logger().info('Goal reached and oriented!')
                return
            else:
                # orient in place
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

        # project to non-holonomic (diff drive): forward vx_b, steer with alpha for lateral vy_b
        if math.hypot(vx_b, vy_b) > 0.01:  # avoid div0
            alpha_b = math.atan2(vy_b, vx_b)
        else:
            alpha_b = 0.0

        # linear forward (clip, avoid backwards if set)
        linear_x = vx_b
        if self.avoid_backwards:
            linear_x = max(linear_x, 0.0)
        linear_x = np.clip(linear_x, 0.0 if self.avoid_backwards else -self.v_max_linear, self.v_max_linear)

        # angular from steering angle (points x-axis towards desired direction)
        angular_z = self.k_ang * alpha_b
        angular_z = np.clip(angular_z, -self.v_max_angular, self.v_max_angular)

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
