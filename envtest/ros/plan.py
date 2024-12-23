import numpy as np
import math
import rospy
from cv_bridge import CvBridge
from mavros_msgs.msg import State
from mavros_msgs.srv import CommandBool, CommandBoolRequest, SetMode, SetModeRequest
from nav_msgs.msg import Odometry, Path
from a_star import AStar
from util_env import Map
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseStamped, Vector3
import time
from scipy.interpolate import BSpline

current_state = State()

def state_cb(msg):
    global current_state
    current_state = msg

camera_matrix = np.array([[386.90203857421875, 0, 319.149169921875], [0, 386.90203857421875, 236.61355590820312], [0, 0, 1]])
cam2body = np.eye(3)
drone_odom = Odometry()
odom_receive = False
need_depth_update = True
env = Map(50, 50, 5)
planner = AStar(goal = np.array([[20, 0, 1.5],[0,0,0]]), env=env)
traj_spline = None

def odom_callback(odom):
	global drone_odom, odom_receive
	drone_odom = odom
	odom_receive = True

def depth_to_point_cloud(depth_image):
	t0 = time.time()
	fx = camera_matrix[0, 0]
	fy = camera_matrix[1, 1]
	cx = camera_matrix[0, 2]
	cy = camera_matrix[1, 2]
	
	height, width = depth_image.shape
	index_x, index_y = np.meshgrid(np.arange(width), np.arange(height))

	Z = depth_image.astype(np.float32)
	X = (index_x - cx) * Z / fx
	Y = (cy - index_y) * Z / fy

	valid = Z < 2
	Z = Z[valid]
	X = X[valid]
	Y = Y[valid]

	points = np.vstack((X.ravel(), Z.ravel(), Y.ravel()))
	# body2world = rowan.to_matrix([odom_msg.pose.pose.orientation.w, odom_msg.pose.pose.orientation.x, odom_msg.pose.pose.orientation.y, odom_msg.pose.pose.orientation.z])
	body2world = np.eye(3)
	cam_pos = np.array([[drone_odom.pose.pose.position.x], [drone_odom.pose.pose.position.y], [drone_odom.pose.pose.position.z]])
	points = body2world @ (cam2body @ points) + cam_pos
	# points = points + cam_pos
	# print(points.shape)
	print(time.time() - t0)
	return points

def image_callback(data):
	global need_depth_update
	if not need_depth_update: return
	depth = bridge.imgmsg_to_cv2(data, "32FC1")
	planner.env.depth = depth
	# print(depth, planner.env.depth)
	# point_cloud = depth_to_point_cloud(depth)
	# planner.env.update_obs_list(point_cloud)
	need_depth_update = False

def pos_publish(msg):
    global traj_spline
    if (traj_spline == None):
        return
    pos_for_publish = PoseStamped()
    print((rospy.Time.now() - last_traj_update).to_sec())
    pos = traj_spline((rospy.Time.now() - last_traj_update).to_sec() / 8)
    print(pos)
    pos_for_publish.pose.position.x = pos[0]
    pos_for_publish.pose.position.y = pos[1]
    pos_for_publish.pose.position.z = pos[2]
    local_pos_pub.publish(pos_for_publish)

def if_reached(target_x: float, target_y: float, target_z: float, threshold: float = 0.20) -> bool:
    current_x = drone_odom.pose.pose.position.x
    current_y = drone_odom.pose.pose.position.y
    current_z = drone_odom.pose.pose.position.z
    distance_x = abs(target_x - current_x)
    distance_y = abs(target_y - current_y)
    distance_z = abs(target_z - current_z)
    distance = math.sqrt(math.pow(distance_x, 2) + math.pow(distance_y, 2) + math.pow(distance_z, 2))
    rospy.loginfo("Reached {}. Distance {}.".format(distance <= threshold, distance))

    return distance <= threshold

if __name__ == '__main__':
    rospy.init_node("offboard_node")
    local_pos_pub = rospy.Publisher("mavros/setpoint_position/local", PoseStamped, queue_size=10)
    rospy.wait_for_service("mavros/cmd/arming")
    arming_client = rospy.ServiceProxy("mavros/cmd/arming", CommandBool)
    rospy.wait_for_service("mavros/set_mode")
    set_mode_client = rospy.ServiceProxy("mavros/set_mode", SetMode)

    bridge = CvBridge()
    rate = rospy.Rate(20)

    state_sub = rospy.Subscriber("mavros/state", State, state_cb)
    odom_sub = rospy.Subscriber("/mavros/local_position/odom", Odometry, odom_callback, queue_size=1, buff_size=256)
    image_sub = rospy.Subscriber("/camera/depth/image_rect_raw", Image, image_callback, queue_size=1, buff_size=2**20)
    pos_pub_timer = rospy.Timer(rospy.Duration(0.1), pos_publish)

    while(rospy.is_shutdown() or not current_state.connected):
        rate.sleep()
    home_position = Vector3()
    target_position = Vector3()
    home_position.x = drone_odom.pose.pose.position.x
    home_position.y = drone_odom.pose.pose.position.y
    home_position.z = drone_odom.pose.pose.position.z

    offb_set_mode = SetModeRequest()
    offb_set_mode.custom_mode = 'OFFBOARD'
    arm_cmd = CommandBoolRequest()
    arm_cmd.value = True

    hover_cnt = 0
    pose_for_publish = PoseStamped()
    mission_step = 0
    last_req = rospy.Time.now()
    home_position_get = False

    while not rospy.is_shutdown():
        if (current_state.mode == "MANUAL" or current_state.mode == "STABILIZED"):
            rospy.loginfo("Mode is MANUAL or STABILIZED, exiting.")
            print("Mode is MANUAL or STABILIZED, exiting.")
            exit()

        if(current_state.mode != 'OFFBOARD' and (rospy.Time.now() - last_req) > rospy.Duration(5.0)):
            if(set_mode_client.call(offb_set_mode).mode_sent == True):
                rospy.loginfo("OFFBOARD enabled")
            last_req = rospy.Time.now()
        else:
            if mission_step == 0:
                rospy.logwarn_once("Mission Step 0.")
                # print("Step 0")
                if (not current_state.armed and (rospy.Time.now() - last_req) > rospy.Duration(5.0)):
                    if (arming_client.call(arm_cmd).success):
                        rospy.loginfo("Vehicle armed")
                        if home_position_get == False:
                            home_position.x = drone_odom.pose.pose.position.x
                            home_position.y = drone_odom.pose.pose.position.y
                            home_position.z = drone_odom.pose.pose.position.z
                            home_position_get = True
                            rospy.loginfo("Home position get: x: {}, y: {}, z: {}".format(
                                home_position.x, home_position.y, home_position.z))
                        
                        mission_step = 1
                    last_req = rospy.Time.now()
            elif mission_step == 1:
                rospy.logwarn_once("Mission Step 1. TAKEOFF and HOVERING.")
                target_position.x = home_position.x
                target_position.y = home_position.y
                target_position.z = home_position.z + 1.0
                pose_for_publish.pose.position.x = target_position.x
                pose_for_publish.pose.position.y = target_position.y
                pose_for_publish.pose.position.z = target_position.z
                local_pos_pub.publish(pose_for_publish)
                if if_reached(target_position.x, target_position.y, target_position.z, threshold=0.5) == True:
                    rospy.loginfo("Target reached. Current position: x: {} y: {} z: {}".format(
                        drone_odom.pose.pose.position.x, drone_odom.pose.pose.position.y, drone_odom.pose.pose.position.z))
                    if hover_cnt == 10:
                        mission_step = 2
                        last_req = rospy.Time.now()
                    hover_cnt += 1
                rospy.loginfo_once("Target Position: x: {} y: {} z: {}".format(target_position.x, target_position.y, target_position.z))
            elif mission_step == 2:
                rospy.logwarn_once("Mission Step 2. Planning.")
                # print("step 2")
                if not odom_receive or need_depth_update:
                    rospy.sleep(0.1)
                    continue
                
                cur_state = np.zeros((2,3))
                cur_state[0,0] = drone_odom.pose.pose.position.x
                cur_state[0,1] = drone_odom.pose.pose.position.y
                cur_state[0,2] = drone_odom.pose.pose.position.z
                cur_state[1,0] = drone_odom.twist.twist.linear.x
                cur_state[1,1] = drone_odom.twist.twist.linear.y
                cur_state[1,2] = drone_odom.twist.twist.linear.z
                t0 = time.time()
                traj = planner.run(cur_state)
                # print("planning time cost", time.time() - t0)
                print(traj)
                last_traj_update = rospy.Time.now()
                traj_pos = np.array([pt[0] for pt in traj])
                print(traj_pos)
                if (len(traj)<=3):
                    traj_spline = BSpline(np.arange(len(traj)+3)-1, traj_pos, 2)
                else:
                    traj_spline = BSpline(np.arange(len(traj)+4)-2, traj_pos, 3)

                rospy.sleep(2)
                need_depth_update = True

                if (rospy.Time.now() - last_traj_update) > rospy.Duration(20.0):
                    mission_step = 3
            elif mission_step == 3:
                rospy.logwarn_once("Mission Step 3. Landing.")
                offb_set_mode.custom_mode = 'AUTO.LAND'
                if (current_state.mode != 'AUTO.LAND' and (rospy.Time.now() - last_req) > rospy.Duration(1.5)):
                    if (set_mode_client.call(offb_set_mode).mode_sent == True):
                        rospy.loginfo("AUTO.LAND enabled")
                        mission_step = 4
                    last_req = rospy.Time.now()
            else:
                rospy.logwarn_once("Mission Completed.")
                break
        rate.sleep()