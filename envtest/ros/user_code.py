#!/usr/bin/python3

from utils import AgileCommandMode, AgileCommand
from cv_bridge import CvBridge
from scipy.interpolate import BSpline

from a_star import AStar
from util_env import Map
from util_node import Node

import numpy as np
import rospy

bridge = CvBridge()
env = Map(50, 50, 5)
planner = AStar(goal=np.array([[60, 0, 1.5], [0, 0, 0]]), env=env)

last_traj_update = 0
traj_spline = None


def compute_command_vision_based(state, img):

    print("state: ", state)

    # update the env
    # print("img: ", img.shape)
    # print("img[1][1]", type(img[1][1]))
    planner.env.depth = img

    # get current state
    cur_state = np.array(
        [
            [state.pos[0], state.pos[1], state.pos[2]],
            [state.vel[0], state.vel[1], state.vel[2]],
        ]
    )

    # plan
    traj = planner.run(cur_state)

    if len(traj) != 0:
        traj_pos = np.array([pt[0] for pt in traj])
        print("traj_pos: ", traj_pos)

        if len(traj) <= 3:
            traj_spline = BSpline(np.arange(len(traj) + 3) - 1, traj_pos, 2)
        else:
            traj_spline = BSpline(np.arange(len(traj) + 4) - 2, traj_pos, 3)

        last_traj_update = rospy.Time.now()

    if traj_spline is not None:
        pos = traj_spline((rospy.Time.now() - last_traj_update).to_sec() / 8)
        velocity = [pos[0] - state.pos[0], pos[1] - state.pos[1], pos[2] - state.pos[2]]
    else:
        velocity = [1.0, 0.0, 0.0]


    # init command
    command_mode = 2
    command = AgileCommand(command_mode)
    command.t = state.t
    command.velocity = velocity
    command.yawrate = 4
    return command


def compute_command_state_based(state, obstacles, rl_policy=None):
    command_mode = 2
    command = AgileCommand(command_mode)
    command.t = state.t
    command.velocity = [1.0, 0.0, 0.0]
    command.yawrate = 10
    return command


def tpl_compute_command_vision_based(state, img):
    ################################################
    # !!! Begin of user code !!!
    # TODO: populate the command message
    ################################################
    print("Computing command vision-based!")
    # print(state)
    # print("Image shape: ", img.shape)

    # Example of SRT command
    command_mode = 0
    command = AgileCommand(command_mode)
    command.t = state.t
    command.rotor_thrusts = [1.0, 1.0, 1.0, 1.0]

    # Example of CTBR command
    command_mode = 1
    command = AgileCommand(command_mode)
    command.t = state.t
    command.collective_thrust = 15.0
    command.bodyrates = [0.0, 0.0, 0.0]

    # Example of LINVEL command (velocity is expressed in world frame)
    command_mode = 2
    command = AgileCommand(command_mode)
    command.t = state.t
    command.velocity = [1.0, 0.0, 0.0]
    command.yawrate = 0.0

    ################################################
    # !!! End of user code !!!
    ################################################

    return command


def tpl_compute_command_state_based(state, obstacles, rl_policy=None):
    ################################################
    # !!! Begin of user code !!!
    # TODO: populate the command message
    ################################################
    print("Computing command based on obstacle information!")
    # print(state)
    # print("Obstacles: ", obstacles)

    # Example of SRT command
    command_mode = 0
    command = AgileCommand(command_mode)
    command.t = state.t
    command.rotor_thrusts = [1.0, 1.0, 1.0, 1.0]

    # Example of CTBR command
    command_mode = 1
    command = AgileCommand(command_mode)
    command.t = state.t
    command.collective_thrust = 10.0
    command.bodyrates = [0.0, 0.0, 0.0]

    # Example of LINVEL command (velocity is expressed in world frame)
    command_mode = 2
    command = AgileCommand(command_mode)
    command.t = state.t
    command.velocity = [1.0, 0.0, 0.0]
    command.yawrate = 0.0

    # If you want to test your RL policy
    # if rl_policy is not None:
    #     command = rl_example(state, obstacles, rl_policy)

    ################################################
    # !!! End of user code !!!
    ################################################

    return command
