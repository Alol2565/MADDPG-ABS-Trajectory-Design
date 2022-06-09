import string

from matplotlib.pyplot import step
from Env.nodes import SP_Node
import numpy as np
from copy import deepcopy
# from logger import log

class UAV(SP_Node):
    def __init__(self, id: string, power: float, initial_location: np.array, env_borders, buildings):
        super().__init__(id, power, initial_location)
        self.velocity = np.array([0., 0., 0])
        self.initial_location = deepcopy(initial_location)
        self.initial_velocity = deepcopy(self.velocity)
        self.max_velocity = np.array([10., 10., 10.])
        self.max_acceleration = np.array([2., 2., 2.])
        self.buildings = buildings
        self.env_borders = env_borders
        self.hit_building = False
        self.collision = False
        self.trajectory = []
        self.trajectory.append(deepcopy(self.location))
        self.battery = 1000
        self.out_of_battery = False

    def collision_occured(self, location):
        for building in self.buildings:
            if(building.obstacle(location)):
                return True
        return False

    def out_of_border(self, location):
        if  (np.any(location < self.env_borders[0]) or np.any(location > self.env_borders[1])):
            return True
        return False

    def valid_location(self, location):
        if (self.out_of_border(location)):
            return False
        return True

    def reset(self):
        self.location = deepcopy(self.initial_location)
        self.power = deepcopy(self.initial_power)
        self.velocity = deepcopy(self.initial_velocity)
        self.trajectory = []
        self.trajectory.append(deepcopy(self.location))
        self.collision = False
        self.battery = 10
        self.out_of_battery = False
        return self

    def move(self, acceleration, delta_time:np.float64):
        """
        Diffrent approach can be used to define the kinematic of UAVs
        """
        # if(self.out_of_battery):
        #     return self
        # self.battery -= self.power
        # if(self.battery < 0):
        #     self.out_of_battery = True
        acceleration = np.array([acceleration[0], acceleration[1], 0])
        self.collision = False

        if(np.any(np.abs(acceleration) > self.max_acceleration)):
            acceleration = self.max_acceleration

        prev_location = deepcopy(self.location)
        self.location += np.multiply(self.velocity, delta_time) + 0.5 * np.multiply(acceleration, delta_time**2)

        if(not self.valid_location(self.location)):
            if(self.location[0] < self.env_borders[0][0] or self.location[0] > self.env_borders[1][0]):
                self.velocity[0] = -self.velocity[0]
                self.location = deepcopy(prev_location)
            if(self.location[1] < self.env_borders[0][1] or self.location[1] > self.env_borders[1][1]):
                self.velocity[1] = -self.velocity[1]
                self.location = deepcopy(prev_location)
            self.collision = True

        prev_velocity = deepcopy(self.velocity)
        self.velocity += np.multiply(acceleration, delta_time)

        if(np.any(np.abs(self.velocity) > self.max_velocity)):
            self.velocity = deepcopy(prev_velocity)
        
        self.trajectory.append(deepcopy(self.location))
        return self
    

    # def move(self, velocity, delta_time:np.float64):
    #     velocity = 5 * np.array([velocity[0], velocity[1], 0])
    #     self.collision = False

    #     if(np.any(np.abs(velocity) > self.max_velocity)):
    #         velocity = self.max_velocity
        
    #     prev_location = deepcopy(self.location)
    #     self.location += np.multiply(self.velocity, delta_time) 

    #     if(not self.valid_location(self.location)):
    #         self.location = deepcopy(prev_location)
    #         self.collision = True
        
    #     self.trajectory.append(deepcopy(self.location))
    #     return self

    