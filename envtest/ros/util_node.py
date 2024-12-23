import numpy as np

# definition of motion primitive
class Node(object):
    def __init__(self, state, parent=None, g=0, h=0, step=0):
        # state is 2*3, position and velocity
        self.state = state
        self.parent = parent
        self.g = g
        self.h = h
        self.step = step
    
    def __eq__(self, node) -> bool:
        return np.all(self.state == node.state)
    
    def __ne__(self, node) -> bool:
        return not self.__eq__(node)

    def __lt__(self, node) -> bool:
        return self.g + self.h < node.g + node.h or (self.g + self.h == node.g + node.h and self.h < node.h)

    def __hash__(self) -> int:
        return hash(self.state)

    def __str__(self) -> str:
        return "parent:{}\ng:{}\nh:{}\nstate:{}".format(self.parent, self.g, self.h, self.state)
    
    @property
    def x(self) -> float:
        return self.state[0]
    @property
    def y(self) -> float:
        return self.current[1]
    @property
    def px(self) -> float:
        if self.parent:
            return self.parent[0]
        else:
            return None
    @property
    def py(self) -> float:
        if self.parent:
            return self.parent[1]
        else:
            return None