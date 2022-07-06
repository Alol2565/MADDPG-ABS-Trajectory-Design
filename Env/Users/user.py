import string
import numpy as np
from Env.nodes import SP_Node, User_Node
from copy import deepcopy

class User(User_Node):
    def __init__(self, id: string, power: float, initial_location, env_borders):
        super().__init__(id, power, initial_location)
        self.location = deepcopy(initial_location)
        self.initial_location =  deepcopy(initial_location)
        self.power = deepcopy(power)
        self.initial_power = deepcopy(self.power)
        self.mean_velocity = 0
        self.sd_velocity = 3
        self.z_scale = 0.00
        self.env_borders = env_borders
        self.request_to_connect = True
        self.connected = False
        self.connected_to:string = 'None'
        self.bit_rate:np.float32 = 0
        self.bit_rate_in_time = []
        self.mean_bit_rate = 0
        self.inside_building = False
        self.current_snr = 0
        self.service_provider:SP_Node = None
        self.trajectory = []
        self.trajectory.append(deepcopy(self.location))

    def __add__(self, other):
        return self.bit_rate + other.bit_rate

    def reset(self):
        self.location = deepcopy(self.initial_location)
        self.power = deepcopy(self.initial_power)
        self.trajectory = []
        self.trajectory.append(deepcopy(self.location))
        self.request_to_connect = True
        self.connected = False
        self.connected_to:string = 'None'
        self.bit_rate:np.float32 = 0
        self.bit_rate_in_time = []
        self.mean_bit_rate = 0
        self.inside_building = False
        self.current_snr = 0
        self.service_provider:SP_Node = None


    def check_instant_rate(self):
        if(self.connected):
            self.bit_rate_in_time.append(self.bit_rate)
            self.mean_bit_rate = np.mean(self.bit_rate_in_time)
            # log.info('%10s is connected to %10s with rate: %20s' %(self.id, self.connected_to, self.bit_rate))
            return self.bit_rate
        self.bit_rate_in_time.append(0)
        self.mean_bit_rate = np.mean(self.bit_rate_in_time)
        return 0

    # def collision_occured(self, location):
    #     for building in self.env.buildings:
    #         if (building.obstacle(location)):
    #             return True
    #     return False
    """
    User can be in building and how they should connect to network???
    """

    def out_of_border(self, location):
        if not (np.all(self.env_borders[0] < location) and np.all([location < self.env_borders[1]])):
            return True
        return False

    def valid_location(self, location):
        if (self.out_of_border(location)):
            return False
        return True

    def move(self, delta_time):
        collision = False
        x_velocity = np.random.normal(self.mean_velocity, self.sd_velocity)
        y_velocity = np.random.normal(self.mean_velocity, self.sd_velocity)
        z_velocity = self.z_scale * np.random.normal(self.mean_velocity, self.sd_velocity)
        self.velocity = np.array([x_velocity, y_velocity, 0], dtype=np.float64)
        prev_location = deepcopy(self.location)
        self.location += self.velocity * delta_time
        if (not self.valid_location(self.location)):
            self.location = deepcopy(prev_location)
            collision = True
        self.trajectory.append(deepcopy(self.location))
        return self
        
        # To - Do
        # Consider different population in different area.
            
        
        
        
        