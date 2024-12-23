import heapq
import copy
from util_env import Map
from util_node import Node
from abc import abstractmethod, ABC
import numpy as np
from scipy.spatial import distance
import time

class AStar(ABC):
    def __init__(self, goal=None, env=None):
        self.goal = Node(goal)
        self.env = env
        self.lower_bound = [-env.x_range/2, -env.y_range/2, 0]
        self.upper_bound = [env.x_range/2, env.y_range/2, 10]
        self.motions = env.motions
        self.tau = 1 # time step
        self.rho = 0.1
        self.danger_dis = 2
        self.stop_dis = 0.5
        self.v_max = 3
    
    def h(self, node, goal):
        dis = np.linalg.norm(node.state[0,:]-goal.state[0,:])
        T = dis / self.v_max
        return dis

    def plan(self, start_node):
        OPEN = []
        heapq.heappush(OPEN, start_node)
        CLOSED = []

        while OPEN:
            node = heapq.heappop(OPEN)
            if np.linalg.norm(node.state[0,:]-self.goal.state[0,:]) < 1e-5 or node.step > 3:
                CLOSED.append(node)
                path = self.extractPath(node, CLOSED)
                self.last_update = time.time()
                return path

            for node_n in self.getNeighbor(node):
                if node_n in CLOSED:
                    continue
                node_n.parent = node.state
                # distance + time + smoothness + distance to obstacle
                node_n.g = node.g + self.rho * self.tau + 0. * np.dot(node_n.state[1,:], node_n.state[1,:]) + 5 * self.dis_cost(node_n)
                # print(node_n, self.dis_cost(node_n))
                # print(self.dis_cost(node_n))
                node_n.h = self.h(node_n, self.goal)
                node_n.step = node.step + 1
                heapq.heappush(OPEN, node_n)
            
            CLOSED.append(node)
        return []
    
    def dis_cost(self, node):
        p = node.state[0, :] + node.state[1,:] * self.tau
        obs_list = self.catch_obs_list(p)
        dis = np.inf
        for obs in obs_list:
            if np.linalg.norm(obs - p) < dis:
                dis = np.linalg.norm(obs - p)
        # for obs in self.env.obs_list:
        #     if np.linalg.norm(obs - p) < dis:
        #         dis = np.linalg.norm(obs - p)
        if dis < self.stop_dis:
            return np.inf
        elif dis < self.danger_dis:
            return 1 / dis
        else:
            return 0
    
    def catch_obs_list(self, p):
        cx = fx = 120
        cy = fy = 80
        p_rel = p - self.start_node.state[0,:]
        if p_rel[1] == 0:
            x, y = 0, 0
        else:
            x = int(round(p_rel[0] * 120 / p_rel[1] + 120))
            y = int(round(-p_rel[2] * 80 / p_rel[1] + 80))
        
        if x-4<0: x=4
        if y-4<0: y=4
        if x+5>120: x=115
        if y+5>80: y=75
        index_x, index_y = np.meshgrid(np.arange(x-4, x+5), np.arange(y-4, y+5))
        Z = self.env.depth[x-4:x+5, y-4:y+5]
        # print(self.env.depth[np.max(X-4,0):X+5, np.max(Y-4,0):Y+5])
        # print(X, Y, Z)
        X = (index_x - cx) * Z / fx
        Y = (cy - index_y) * Z / fy
        points = np.vstack((X.ravel(), Z.ravel(), Y.ravel())).transpose()
        return points + self.start_node.state[0,:]

    def isCollision(self, node):
        # check if outside the environment
        if np.any(node.state[0,:] < self.lower_bound) or np.any(node.state[0,:] > self.upper_bound):
            return True
        p = node.state[0, :] + node.state[1, :] * self.tau
        obs_list = self.catch_obs_list(p)

        # else:
        for obs in obs_list:
            if distance.euclidean(obs, p) < self.stop_dis:
                return True
        # for obs in self.env.obs_list:
        #     if distance.euclidean(obs, p) < self.stop_dis:
        #         return True
        return False

    def getNeighbor(self, node):
        ret = []
        neighbor = copy.deepcopy(node)
        for motion in self.motions:
            neighbor.state[0,:] = node.state[0,:] + self.tau * node.state[1,:]
            # vel = np.sqrt(node.state[1,0]**2+node.state[1,1]**2)
            # angle = np.arctan2(node.state[1,1], node.state[1,0])
            # neighbor.state[1,0] = np.cos(angle + motion[1]/180*np.pi) * (vel + motion[0] * self.tau)
            # neighbor.state[1,1] = np.sin(angle + motion[1]/180*np.pi) * (vel + motion[0] * self.tau)
            neighbor.state[1,:] = node.state[1,:] + self.tau * np.array(motion)
            if not self.isCollision(neighbor):
                ret.append(copy.deepcopy(neighbor))
        return ret

    def extractPath(self, end_node, closed):
        node = closed[closed.index(end_node)]
        path = [node.state]
        while not np.all(node.state[0,:] == self.start_node.state[0,:]):
            node_parent = closed[closed.index(Node(node.parent))]
            node = node_parent
            path.append(node.state)
        return path[::-1]

    def run(self, state):
        self.start_node = Node(state)
        # self.start_node.state[1,1] = 1
        path = self.plan(self.start_node)
        return path
        
