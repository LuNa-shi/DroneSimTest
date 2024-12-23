from abc import ABC
import numpy as np

class Map(ABC):
    def __init__(self, x_range, y_range, z_range) -> None:
        self.x_range = x_range
        self.y_range = y_range
        self.z_range = z_range
        self.obs_list = []
        self.motions = []
        self.depth = np.zeros((240, 160))
        self.cam = np.array([[120, 0, 120], [0, 80, 80], [0, 0, 1]])

        # list all available actions. 
        # (horizontal speed change, horizontal angle change, vertical speed change)
        for i in [-1, 0, 1]:
            for j in [0, 1]:
                for k in [0]:
                    if (i==0 and j==0 and k==0): continue
                    motion = (i, j, k)
                    self.motions.append(motion)
    
    def update_obs_list(self, obs_list):
        self.obs_list = obs_list.transpose()
    
