from copy import deepcopy
import string
from numpy import power
from Env.nodes import SP_Node
import numpy as np
# from logger import log

class Base_Station(SP_Node):
    def __init__(self, id: string, power: float, initial_location: np.array):
        super().__init__(id, power, initial_location)
        self.id = deepcopy(id)
        self.power = deepcopy(power)
        self.location = deepcopy(initial_location)
        self.users = [] 
        

    def reset(self):
        self.users = []
        return super().reset()
        
    ### To Do 
    ### 1. Assign some UAVs and users to it.
    ### 2. Define it's kind and extract some feature to it.
    