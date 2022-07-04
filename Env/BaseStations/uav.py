import string
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
        self.max_velocity = np.array([9., 9., 9.])
        self.max_acceleration = np.array([1., 1., 0.1])
        self.buildings = buildings
        self.env_borders = env_borders
        self.hit_building = False
        self.collision = False
        self.trajectory = []
        self.trajectory.append(deepcopy(self.location))
        self.battery = 1000
        self.out_of_battery = False
        self.users = []

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
        self.users = []
        return self

    def move(self, angle, delta_time:np.float64):
        angle = angle[0]
        if(angle > np.pi or angle < -np.pi):
            angle = 0
        self.velocity = np.multiply(np.array([np.cos(angle),np.sin(angle), 0]), 8)
        prev_location = deepcopy(self.location)
        self.location += np.multiply(self.velocity, delta_time)

        if(not self.valid_location(self.location)):
            if(self.location[0] < self.env_borders[0][0] or self.location[0] > self.env_borders[1][0]):
                self.velocity[0] = -self.velocity[0]
                self.location = deepcopy(prev_location)
            if(self.location[1] < self.env_borders[0][1] or self.location[1] > self.env_borders[1][1]):
                self.velocity[1] = -self.velocity[1]
                self.location = deepcopy(prev_location)
            self.collision = True   
        self.trajectory.append(deepcopy(self.location))
        return self


    # def move(self, acceleration, delta_time:np.float64):
    #     """
    #     Diffrent approach can be used to define the kinematic of UAVs
    #     """
    #     # if(self.out_of_battery):
    #     #     return self
    #     # self.battery -= self.power
    #     # if(self.battery < 0):
    #     #     self.out_of_battery = True
    #     acceleration = np.array([acceleration[0], acceleration[1], 0])
    #     self.collision = False


    #     for i in range(len(acceleration)):
    #         if(acceleration[i] > self.max_acceleration[i]):
    #             acceleration[i] = self.max_acceleration[i]
    #         elif(acceleration[i] < -self.max_acceleration[i]):
    #             acceleration[i] = -self.max_acceleration[i]

    #     prev_location = deepcopy(self.location)
    #     self.location += np.multiply(self.velocity, delta_time) + 0.5 * np.multiply(acceleration, delta_time**2)

    #     if(not self.valid_location(self.location)):
    #         if(self.location[0] < self.env_borders[0][0] or self.location[0] > self.env_borders[1][0]):
    #             self.velocity[0] = -self.velocity[0]
    #             self.location = deepcopy(prev_location)
    #         if(self.location[1] < self.env_borders[0][1] or self.location[1] > self.env_borders[1][1]):
    #             self.velocity[1] = -self.velocity[1]
    #             self.location = deepcopy(prev_location)
    #         self.collision = True

    #     self.velocity += np.multiply(acceleration, delta_time)

    #     for i in range(len(self.velocity)):
    #         if(self.velocity[i] > self.max_velocity[i]):
    #             self.velocity[i] = self.max_velocity[i]
    #         elif(self.velocity[i] < -self.max_velocity[i]):
    #             self.velocity[i] = -self.max_velocity[i]
        
    #     self.trajectory.append(deepcopy(self.location))
    #     return self
    