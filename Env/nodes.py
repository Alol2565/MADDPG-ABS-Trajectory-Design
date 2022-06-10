import string
import numpy as np
from copy import deepcopy
# from logger import log

class SP_Node:
    def __init__(self, id:string, power:float, initial_location:np.array):
        self.id = deepcopy(id)
        self.initial_location = deepcopy(initial_location)
        self.location = deepcopy(self.initial_location)
        self.initial_power = deepcopy(power)
        self.power = deepcopy(self.initial_power)
        self.users = []
        

    def reset(self):
        self.location = deepcopy(self.initial_location)
        self.power = deepcopy(self.initial_power)
        self.users = []

    def append_user(self, user):
        self.users.append(user)
        # log.info('%s connected to %s' %(user.id, self.id))
        return True

    def remove_user(self, user):
        for i in range(len(self.users)):
            if(self.users[i].id == user.id):
                self.users.pop(i)
                # log.info('%7s has been removed from %7s' %(user.id, self.id))
                return True
        return False

class User_Node:
    def __init__(self, id:string, power:float, initial_location:np.array):
        self.id = deepcopy(id)
        self.initial_location = deepcopy(initial_location)
        self.location = deepcopy(self.initial_location)
        self.initial_power = deepcopy(power)
        self.power = deepcopy(self.initial_power)
        self.service_provider:SP_Node
        self.snr:float = 0

    def reset(self):
        self.location = deepcopy(self.initial_location)
        self.power = deepcopy(self.initial_power)

    # def refresh_network(self):
    #     if(self.service_provider == None):
    #         self.connect_bset_network()
    #     self.disconnect_network()
    #     self.connect_bset_network()

