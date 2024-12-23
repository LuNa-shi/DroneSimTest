#!/usr/bin/python3

from utils import AgileCommandMode, AgileCommand
from cv_bridge import CvBridge
from scipy.interpolate import BSpline

from a_star import AStar
from util_env import Map
from util_node import Node

import numpy as np
import rospy

planner = AStar(goal=np.array([[60, 0, 1.5], [0, 0, 0]]), env=Map(50, 50, 10))

last_traj_update = 0
traj_spline = None


def compute_command_vision_based(state, img):

    # get current state
    cur_state = np.array(
        [
            [state.pos[0], state.pos[1], state.pos[2]],
            [state.vel[0], state.vel[1], state.vel[2]],
        ]
    )

    # plan
    planner.env.depth = img
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
        velocity = state.vel

    # command
    command_mode = 2
    command = AgileCommand(command_mode)
    command.t = state.t
    command.velocity = velocity
    command.yawrate = 0
    return command


def compute_command_state_based(state, obstacles, rl_policy=None):
    command_mode = 2
    command = AgileCommand(command_mode)
    command.t = state.t
    command.velocity = [1.0, 0.0, 0.0]
    command.yawrate = 10
    return command
