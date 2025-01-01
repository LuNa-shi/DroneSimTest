#!/usr/bin/env python3
import collections
import copy
import os
import numpy as np
import rospy
import cv2
import pandas as pd
import datetime
import shutil
import math

from dodgeros_msgs.msg import QuadState
from cv_bridge import CvBridge, CvBridgeError
from nav_msgs.msg import Odometry
from geometry_msgs.msg import TwistStamped
# from quadrotor_msgs.msg import TrajectoryPoint
# from quadrotor_msgs.msg import Trajectory
from sensor_msgs.msg import Image
from std_msgs.msg import Bool
from std_msgs.msg import Empty
from scipy.spatial.transform import Rotation as R
# from agile_autonomy_msgs.msg import MultiTrajectory
from minsnap import Point, TrajectoryExt
import tensorflow as tf
import yaml

from models.plan_learner import PlanLearner



class MultiTrajectory:
    def __init__(self):
        self.header = None
        self.ref_pose = None
        self.trajectories = []
        self.execute = False





class PlanBase(object):
    def __init__(self, config, mode):
        print("Initializing agile_pilot_node...")
        rospy.init_node('agile_pilot_node', anonymous=False)

        self.vision_based = True
        self.publish_commands = False
        self.cv_bridge = CvBridge()

        self.config = config
        self.odometry = Odometry()
        self.gt_odometry = Odometry()
        self.maneuver_complete = False
        self.use_network = False
        self.net_initialized = False
        self.reference_initialized = False
        self.rollout_idx = 0
        self.odometry_used_for_inference = None
        self.time_prediction = None
        self.last_depth_received = rospy.Time.now()
        self.reference_progress = 0
        self.reference_len = 1
        self.bridge = CvBridge()
        self.mode = mode
        self.image = np.zeros(
            (self.config.img_height, self.config.img_width, 3))
        self.depth = np.zeros(
            (self.config.img_height, self.config.img_width, 3))
        self.n_times_expert = 0
        self.n_times_net = 0.00001
        self.start_time = None
        self.quad_name = config.quad_name
        self.depth_topic = config.depth_topic
        self.rgb_topic = config.rgb_topic
        self.odometry_topic = config.odometry_topic
        self.learner = PlanLearner(settings=config)
        # Queue Stuff
        self.img_queue = collections.deque([], maxlen=self.config.input_update_freq)
        self.depth_queue = collections.deque([], maxlen=self.config.input_update_freq)
        self.state_queue = collections.deque([], maxlen=self.config.input_update_freq)
        self.reset_queue()
        # Init Network
        self._prepare_net_inputs()
        _ = self.learner.inference(self.net_inputs)
        print("Net initialized")
        self.net_initialized = True
        if self.mode == 'training':
            return  # Nothing to initialize
        
        self.start_sub = rospy.Subscriber("/" + self.quad_name + "/start_navigation", Empty, self.callback_fly,
                                          queue_size=1, tcp_nodelay=True)

        self.ground_truth_odom = rospy.Subscriber("/" + self.quad_name + "/" + self.odometry_topic,
                                                  QuadState,
                                                  self.callback_gt_odometry,
                                                  queue_size=1)
        # if self.config.use_rgb:
        #     self.image_sub = rospy.Subscriber("/" + self.quad_name + "/" + self.rgb_topic, Image,
        #                                       self.callback_image, queue_size=1)
        if self.config.use_depth:
            self.depth_sub = rospy.Subscriber("/" + self.quad_name + "/" + self.depth_topic, Image,
                                              self.callback_depth, queue_size=1)

        self.linvel_pub = rospy.Publisher("/" + self.quad_name + "/dodgeros_pilot/velocity_command", TwistStamped,
                                          queue_size=1)
        
        # self.traj_pub = rospy.Publisher("/{}/trajectory_predicted".format(self.quad_name), MultiTrajectory,
        #                                     queue_size=1)  # Stop upon some condition
        self.timer_input = rospy.Timer(rospy.Duration(1. / self.config.input_update_freq),
                                       self.update_input_queues)
        self.timer_net = rospy.Timer(rospy.Duration(1. / self.config.network_frequency),
                                     self._generate_plan)

    def load_trajectory(self, traj_fname):
        self.reference_initialized = False
        traj_df = pd.read_csv(traj_fname, delimiter=',')
        self.reference_len = traj_df.shape[0]
        self.full_reference = Trajectory()
        time = ['time_from_start']
        time_values = traj_df[time].values
        pos = ["pos_x", "pos_y", "pos_z"]
        pos_values = traj_df[pos].values
        vel = ["vel_x", "vel_y", "vel_z"]
        vel_values = traj_df[vel].values
        for i in range(self.reference_len):
            point = TrajectoryPoint()
            point.time_from_start = rospy.Duration(time_values[i])
            point.pose.position.x = pos_values[i][0]
            point.pose.position.y = pos_values[i][1]
            point.pose.position.z = pos_values[i][2]
            point.velocity.linear.x = vel_values[i][0]
            point.velocity.linear.y = vel_values[i][1]
            point.velocity.linear.z = vel_values[i][2]
            self.full_reference.points.append(point)
        # Change type for easier use
        self.full_reference = self.full_reference.points
        self.reference_progress = 0
        self.reference_initialized = True
        assert len(self.full_reference) == self.reference_len
        print("Loaded traj {} with {} elems".format(
            traj_fname, self.reference_len))
        return

    def update_reference_progress(self, quad_position):
        reference_point = self.full_reference[self.reference_progress]
        reference_position_wf = np.array([reference_point.pose.position.x,
                                          reference_point.pose.position.y,
                                          reference_point.pose.position.z]).reshape((3, 1))
        distance = np.linalg.norm(reference_position_wf - quad_position)
        for k in range(self.reference_progress + 1, self.reference_len):
            reference_point = self.full_reference[k]
            reference_position_wf = np.array([reference_point.pose.position.x,
                                              reference_point.pose.position.y,
                                              reference_point.pose.position.z]).reshape((3, 1))
            next_point_distance = np.linalg.norm(reference_position_wf - quad_position)
            if next_point_distance > distance:
                break
            else:
                self.reference_progress = k
                distance = next_point_distance

    def callback_fly(self):
        # Load the trajectory at start
        self.reference_initialized = False
        print("Start publishing commands!")
        self.publish_commands = True
        # if data.data:
        #     # Load the reference trajectory
        #     rollout_dir = os.path.join(self.config.expert_folder,
        #                                sorted(os.listdir(self.config.expert_folder))[-1])
        #     # Load the reference trajectory
        #     if self.config.track_global_traj:
        #         traj_fname = os.path.join(rollout_dir, "ellipsoid_trajectory.csv")
        #     else:
        #         traj_fname = os.path.join(rollout_dir, "reference_trajectory.csv")
        #     print("Reading Trajectory from %s" % traj_fname)
        #     self.load_trajectory(traj_fname)
        #     self.reference_initialized = True
            # Learning phase to test
        tf.keras.backend.set_learning_phase(0)

        # Might be here if you crash in less than a second.
        if self.maneuver_complete:
            return
        # If true, network should fly.
        # If false, maneuver is finished and network is off.
        # self.use_network = data.data and self.config.execute_nw_predictions
        self.use_network = self.config.execute_nw_predictions
        # if (not data.data):
        #     self.maneuver_complete = True
        #     self.use_network = False

    # def experiment_report(self):
    #     print("experiment done")
    #     metrics_report = {}
    #     metrics_report['expert_usage'] = self.n_times_expert / \
    #                                      (self.n_times_net + self.n_times_expert) * 100.
    #     return metrics_report

    # def callback_land(self, data):
    #     self.config.execute_nw_predictions = False

    # def preprocess_img(self, img):
    #     dim = (self.config.img_width, self.config.img_height)
    #     img = cv2.resize(img, dim)
    #     img = np.array(img, dtype=np.float32)
    #     return img

    # def callback_image(self, data):
    #     '''
    #     Reads an image and generates a new plan.
    #     '''
    #     try:
    #         image = self.bridge.imgmsg_to_cv2(data, "bgr8")
    #         if self.quad_name == 'hawk':
    #             image = cv2.flip(image, -1)  # passing a negative axis index flips both axes
    #         if np.sum(image) != 0:
    #             self.image = self.preprocess_img(image)
    #     except CvBridgeError as e:
    #         print(e)

    def preprocess_depth(self, depth):
        depth = np.minimum(depth, 20000)
        dim = (self.config.img_width, self.config.img_height)
        depth = cv2.resize(depth, dim)
        depth = np.array(depth, dtype=np.float32)
        depth = depth / (80) # normalization factor to put depth in (0,255)
        depth = np.expand_dims(depth, axis=-1)
        depth = np.tile(depth, (1, 1, 3))
        return depth

    def callback_depth(self, data):
        '''
        Reads a depth image and saves it.
        '''
        try:
            if self.quad_name == 'kingfisher':
                depth = self.bridge.imgmsg_to_cv2(data, '16UC1')
                # print("============================================================")
                # print("Min Depth {}. Max Depth {}. with Nans {}".format(np.min(depth),
                #                                                        np.max(depth),
                #                                                        np.any(np.isnan(depth))))
                # print("Min Depth {}. Max Depth {}. with Nans {}".format(np.min(depth),
                #                                                        np.max(depth),
                #                                                        np.any(np.isnan(depth))))
            elif self.quad_name == 'hawk':
                # print("Received depth image from hawk!")
                # the depth sensor is mounted in a flipped configuration on the quadrotor, so flip image first
                depth_flipped = self.bridge.imgmsg_to_cv2(data, '16UC1')
                depth = cv2.flip(depth_flipped, -1)  # passing a negative axis index flips both axes
            else:
                print("Invalid quad_name!")
                raise NotImplementedError
            if (np.sum(depth) != 0) and (not np.any(np.isnan(depth))):
                self.depth = self.preprocess_depth(depth)
                self.last_depth_received = rospy.Time.now()

        except CvBridgeError as e:
            print(e)

    def reset_queue(self):
        self.img_queue.clear()
        self.depth_queue.clear()
        self.state_queue.clear()
        self.odom_rot = [0 for _ in range(9)]
        n_init_states = 21
        for _ in range(self.config.input_update_freq):
            self.img_queue.append(np.zeros_like(self.image))
            self.depth_queue.append(np.zeros_like(self.depth))
            self.state_queue.append(np.zeros((n_init_states,)))

    # def callback_start(self, data):
    #     print("Callback START")
    #     self.pipeline_off = False

    # def callback_off(self, data):
    #     print("Callback OFF")
    #     self.pipeline_off = True

    # def maneuver_finished(self):
    #     return self.maneuver_complete

    def quadState_to_odometry(self, quad_state):
        #quadState contains:
        #  Time [s]
        # float64 t
        # # Pose
        # geometry_msgs/Pose pose
        # # Linear & angular velocity
        # geometry_msgs/Twist velocity
        # # Linear & angular acceleration
        # geometry_msgs/Twist 
        odometry = Odometry()
        odometry.header.stamp = quad_state.t
        odometry.pose.pose.position.x = quad_state.pose.position.x
        odometry.pose.pose.position.y = quad_state.pose.position.y
        odometry.pose.pose.position.z = quad_state.pose.position.z
        odometry.pose.pose.orientation.x = quad_state.pose.orientation.x
        odometry.pose.pose.orientation.y = quad_state.pose.orientation.y
        odometry.pose.pose.orientation.z = quad_state.pose.orientation.z
        odometry.pose.pose.orientation.w = quad_state.pose.orientation.w
        odometry.twist.twist.linear.x = quad_state.velocity.linear.x
        odometry.twist.twist.linear.y = quad_state.velocity.linear.y
        odometry.twist.twist.linear.z = quad_state.velocity.linear.z
        odometry.twist.twist.angular.x = quad_state.velocity.angular.x
        odometry.twist.twist.angular.y = quad_state.velocity.angular.y
        odometry.twist.twist.angular.z = quad_state.velocity.angular.z
        return odometry





    def callback_gt_odometry(self, data):
        
        odometry = self.quadState_to_odometry(data)
        rot_body = R.from_quat([odometry.pose.pose.orientation.x,
                                odometry.pose.pose.orientation.y,
                                odometry.pose.pose.orientation.z,
                                odometry.pose.pose.orientation.w])

        R_body_cam = R.from_quat([0.0,
                                  np.sin(-self.config.pitch_angle / 2.0),
                                  0.0,
                                  np.cos(-self.config.pitch_angle / 2.0)])
        rot_cam = rot_body * R_body_cam

        v_B = np.array([odometry.twist.twist.linear.x,
                        odometry.twist.twist.linear.y,
                        odometry.twist.twist.linear.z]).reshape((3, 1))
        w_B = np.array([odometry.twist.twist.angular.x,
                        odometry.twist.twist.angular.y,
                        odometry.twist.twist.angular.z]).reshape((3, 1))
        v_C = R_body_cam.as_matrix().T @ v_B
        w_C = R_body_cam.as_matrix().T @ w_B

        odometry.twist.twist.linear.x = v_C[0]
        odometry.twist.twist.linear.y = v_C[1]
        odometry.twist.twist.linear.z = v_C[2]
        odometry.twist.twist.angular.x = w_C[0]
        odometry.twist.twist.angular.y = w_C[1]
        odometry.twist.twist.angular.z = w_C[2]
        rot_init_body = rot_cam
        self.odom_rot_input = rot_init_body.as_matrix().reshape((9,)).tolist()
        self.odom_rot = rot_cam.as_matrix().reshape((9,)).tolist()


        if self.config.velocity_frame == 'wf':
            # Convert velocity in world frame
            R_W_C = np.array(self.odom_rot).reshape((3, 3))
            v_C = np.array([odometry.twist.twist.linear.x,
                            odometry.twist.twist.linear.y,
                            odometry.twist.twist.linear.z]).reshape((3, 1))
            v_W = R_W_C @ v_C
            odometry.twist.twist.linear.x = v_W[0]
            odometry.twist.twist.linear.y = v_W[1]
            odometry.twist.twist.linear.z = v_W[2]
        self.odometry = odometry


    def _convert_to_traj(self, net_prediction):
        net_prediction = np.reshape(net_prediction, ((self.config.state_dim, self.config.out_seq_len)))
        pred_traj = TrajectoryExt()
        sample_time = np.arange(start=0.1, stop=self.config.out_seq_len) / 10.0
        for k, t in enumerate(sample_time):
            #initialize a 3d point
            position = np.ndarray((3,), dtype=np.float32)
            point = Point(position=position)
            # point.heading = 0.0
            point.time_from_start = t
            point.position[0]= net_prediction[0, k]
            point.position[1]= net_prediction[1, k]
            point.position[2] = net_prediction[2, k]
            pred_traj.points.append(point)
        pred_traj.set_frame('world')
        return pred_traj

    def update_input_queues(self, data):
        # Positions are ignored in the new network
        imu_states = [self.odometry.pose.pose.position.x,
                      self.odometry.pose.pose.position.y,
                      self.odometry.pose.pose.position.z] + \
                     self.odom_rot_input

        vel = np.array([self.odometry.twist.twist.linear.x,
                        self.odometry.twist.twist.linear.y,
                        self.odometry.twist.twist.linear.z])

        #vel = vel / np.linalg.norm(vel) * 7.
        vel = vel.squeeze()
        imu_states = imu_states + vel.tolist()

        if self.config.use_bodyrates:
            imu_states.extend([self.odometry.twist.twist.angular.x,
                               self.odometry.twist.twist.angular.y,
                               self.odometry.twist.twist.angular.z])

        if self.reference_initialized:
            quad_position = np.array([self.odometry.pose.pose.position.x,
                                      self.odometry.pose.pose.position.y,
                                      self.odometry.pose.pose.position.z]).reshape((3, 1))
            self.update_reference_progress(quad_position)
            ref_idx = np.minimum(self.reference_progress +
                                 int(self.config.future_time*50), self.reference_len - 1)
        else:
            ref_idx = 0
        if self.reference_initialized:
            reference_point = self.full_reference[ref_idx]
            reference_position_wf = np.array([reference_point.pose.position.x,
                                              reference_point.pose.position.y,
                                              reference_point.pose.position.z]).reshape((3, 1))
            current_position_wf = np.array([self.odometry.pose.pose.position.x,
                                         self.odometry.pose.pose.position.y,
                                         self.odometry.pose.pose.position.z]).reshape((3, 1))
            difference = reference_position_wf - current_position_wf
            difference = difference / np.linalg.norm(difference)
            goal_dir = self.adapt_reference(difference)
        else:
            # Reference is not loaded at init, but we want to keep updating the list anyway
            goal_dir = np.zeros((3, 1))
        goal_dir = np.squeeze(goal_dir).tolist()

        state_inputs = imu_states + goal_dir
        self.state_queue.append(state_inputs)
        # Prepare images
        if self.config.use_rgb:
            self.img_queue.append(self.image)
        if self.config.use_depth:
            self.depth_queue.append(self.depth)

    #will not used
    def adapt_reference(self, goal_dir):
        if self.config.velocity_frame == 'wf':
            return goal_dir
        elif self.config.velocity_frame == 'bf':
            R_W_C = np.array(self.odom_rot).reshape((3, 3))
            v_C = R_W_C.T @ goal_dir
            return v_C
        else:
            raise IOError("Reference frame not recognized")

    def select_inputs_in_freq(self, input_list):
        new_list = []
        for i in self.required_elements:
            new_list.append(input_list[i])
        return new_list

    def _prepare_net_inputs(self):
        if not self.net_initialized:
            # prepare the elements that need to be fetched in the list
            required_elements = np.arange(start=0, stop=self.config.input_update_freq,
                                          step=int(np.ceil(self.config.input_update_freq / self.config.seq_len)),
                                          dtype=np.int64)
            required_elements = -1 * (required_elements + 1)  # we need to take things at the end :)
            self.required_elements = [i for i in reversed(required_elements.tolist())]
            # return fake input for init
            if self.config.use_bodyrates:
                n_init_states = 21
            else:
                n_init_states = 18
            inputs = {'rgb': np.zeros((1, self.config.seq_len, self.config.img_height, self.config.img_width, 3),
                                      dtype=np.float32),
                      'depth': np.zeros((1, self.config.seq_len, self.config.img_height, self.config.img_width, 3),
                                        dtype=np.float32),
                      'imu': np.zeros((1, self.config.seq_len, n_init_states), dtype=np.float32)}
            self.net_inputs = inputs
            return
        state_inputs = np.stack(self.select_inputs_in_freq(self.state_queue), axis=0)
        state_inputs = np.array(state_inputs, dtype=np.float32)
        new_dict = {'imu': np.expand_dims(state_inputs, axis=0)}
        if self.config.use_rgb:
            img_inputs = np.stack(self.select_inputs_in_freq(self.img_queue), axis=0)
            img_inputs = np.array(img_inputs, dtype=np.float32)
            new_dict['rgb'] = np.expand_dims(img_inputs, axis=0)
        if self.config.use_depth:
            depth_inputs = np.stack(self.select_inputs_in_freq(self.depth_queue), axis=0)
            depth_inputs = np.array(depth_inputs, dtype=np.float32)
            new_dict['depth'] = np.expand_dims(depth_inputs, axis=0)
        self.odometry_used_for_inference = copy.deepcopy(self.odometry)
        self.time_prediction = rospy.Time.now()
        self.net_inputs = new_dict

    def evaluate_dagger_condition(self):
        # Real world dagger condition is only based on time
        if self.reference_progress < 50:
            # At the beginning always use expert (otherwise gives problems)
            print("Expert warm up!")
            return False
        else:
            print("Network in action")
            return True

    def trajectory_decision(self, net_predictions):
        net_in_control = self.evaluate_dagger_condition()
        # select best traj
        # multi_traj = MultiTrajectory()
        # multi_traj.execute = net_in_control
        # multi_traj.header.stamp = self.time_prediction
        # multi_traj.ref_pose = self.odometry_used_for_inference.pose.pose
        best_alpha = net_predictions[0][0]
        traj = TrajectoryExt()
        for i in range(self.config.modes):
            traj_pred = net_predictions[1][i]
            alpha = net_predictions[0][i]
            # convert in a traj
            # if i == 0 or (best_alpha / alpha > self.config.accept_thresh):
            if alpha == best_alpha:
                traj = self._convert_to_traj(traj_pred)
                break
                # multi_traj.trajectories.append(traj_pred)
        #TODO : post process the trajectory 

        # traj.resample_points_from_poly_coeffs()
        startTime= traj.points[0].time_from_start
        for point in traj.points:
            point.time_from_start -= startTime

        for point in traj.points:
            print(point.time_from_start)  
        traj.fit_polynomial_coeffs(5,2)
        dt = 0.1
        vel = traj.evaluate_poly(0.1, 1)
        twist = TwistStamped()
        twist.header.stamp = rospy.Time.now()
        twist.twist.linear.x = vel[0]
        twist.twist.linear.y = vel[1]
        twist.twist.linear.z = vel[2]
        twist.twist.angular.x = 0.0
        twist.twist.angular.y = 0.0
        twist.twist.angular.z = 0.0
        self.linvel_pub.publish(twist)



        # self.traj_pub.publish(multi_traj)
        if net_in_control:
            self.n_times_net += 1
        else:
            self.n_times_expert += 1

    def _generate_plan(self, _timer):
        if (self.image is None) or \
                (not self.net_initialized) or (not self.config.perform_inference):
                # (not self.reference_initialized) or 
                
            return
        t_start = rospy.Time.now()
        if (t_start - self.last_depth_received).to_sec() > 2.0:
            print("Stopping because no depth received")
            self.config.perform_inference = False
            self.callback_land(Empty())

        self._prepare_net_inputs()
        results = self.learner.inference(self.net_inputs)
        self.trajectory_decision(results)

def create_settings(settings_yaml, mode='test'):
    setting_dict = {'test': TestSetting}                    
    settings = setting_dict.get(mode, None)
    if settings is None:
        raise IOError("Unidentified Settings")
    settings = settings(settings_yaml)
    if mode == 'test' or mode == 'openloop':
        settings.freeze_backbone = True
    return settings


class Settings:
    def __init__(self, settings_yaml, generate_log=True):
        assert os.path.isfile(settings_yaml), settings_yaml

        with open(settings_yaml, 'r') as stream:
            settings = yaml.safe_load(stream)

            self.quad_name = settings['quad_name']
            self.odometry_topic = settings['odometry_topic']
            self.rgb_topic = settings['rgb_topic']
            self.depth_topic = settings['depth_topic']
            # Input mode
            self.use_rgb = settings['use_rgb']
            self.use_depth = settings['use_depth']
            self.img_width = settings['img_width']
            self.img_height = settings['img_height']
            self.future_time = settings['future_time']
            # Output config
            self.state_dim = settings['state_dim']
            self.out_seq_len = settings['out_seq_len']
            self.predict_state_number = settings['predict_state_number']
            self.modes = settings['modes']
            self.seq_len = settings['seq_len']
            # net inputs
            inputs = settings['inputs']
            self.use_position = inputs['position']
            self.use_attitude = inputs['attitude']
            self.use_bodyrates = inputs['bodyrates']
            self.velocity_frame = inputs['velocity_frame']

            # --- checkpoint ---
            checkpoint = settings['checkpoint']
            self.resume_training = checkpoint['resume_training']
            assert isinstance(self.resume_training, bool)
            self.resume_ckpt_file = checkpoint['resume_file']

            # Save a copy of the parameters for reproducibility
            log_root = settings['log_dir']
            if not log_root == '' and generate_log:
                current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
                self.log_dir = os.path.join(log_root, current_time)
                os.makedirs(self.log_dir)
                net_file = "./models/nets.py"
                assert os.path.isfile(net_file)
                shutil.copy(net_file, self.log_dir)
                shutil.copy(settings_yaml, self.log_dir)

    def add_flags(self):
        self._add_flags()

    def _add_flags(self):
        raise NotImplementedError

class TestSetting(Settings):
    def __init__(self, settings_yaml):
        super(TestSetting, self).__init__(settings_yaml, generate_log=True)
        self.settings_yaml = settings_yaml
        self.add_flags()

    def _add_flags(self):
        with open(self.settings_yaml, 'r') as stream:
            settings = yaml.safe_load(stream)
            self.ref_frame = settings['ref_frame']
            # special input
            inputs = settings['inputs']
            self.pitch_angle = inputs['pitch_angle'] / 180.0 * math.pi
            test_time = settings['test_time']
            self.execute_nw_predictions = test_time['execute_nw_predictions']
            self.perform_inference = test_time['perform_inference']
            assert isinstance(self.execute_nw_predictions, bool)
            self.max_rollouts = test_time['max_rollouts']
            self.expert_folder = test_time['expert_folder']
            self.crashed_thr = test_time['crashed_thr']
            # Prediction speed
            self.network_frequency = test_time['network_frequency']
            self.fallback_radius_expert = test_time['fallback_radius_expert']
            self.accept_thresh = test_time['accept_thresh']
            self.input_update_freq = test_time['input_update_freq']
            # spacings
            self.tree_spacings = test_time['spacings']
            self.verbose = settings['verbose']
            assert isinstance(self.verbose, bool)
            self.track_global_traj = test_time['track_global_traj']
            # Unity
            unity = settings['unity']
            self.unity_start_pos = unity['unity_start_pos']
            self.random_seed = unity['random_seed']

if __name__ == '__main__':
    config_path = "../../config/test_settings.yaml"
    mode = 'test'
    config = create_settings(config_path, mode)
    plan = PlanBase(config, mode)
    rospy.spin()