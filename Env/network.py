from typing import List
from Env.channel import *
import numpy as np
# from logger import log
from Env.Users.user import User
from Env.BaseStations.uav import UAV
from Env.BaseStations.base_station import Base_Station
from Env.nodes import SP_Node

"""
To Do List:
- [ ] render visual topology
"""


class Network:
    def __init__(self, f_c, eta, sigma, users=[], BSs=[], uavs=[], buildings=[]):
        self.base_stations: List[Base_Station] = BSs
        self.uavs: List[UAV] = uavs
        self.users: List[User] = users
        self.buildings: List[Building] = buildings
        self.f_c = f_c
        self.eta = eta
        self.sigma = sigma

    def register_base_station(self, bs):
        self.base_stations.append(bs)

    def register_abs(self, uav):
        self.uavs.append(uav)

    def register_user(self, user):
        self.users.append(user)

    def find_node(self, node_id):
        for user in self.users:
            if(user.id == node_id):
                return user

        for bs in self.base_stations:
            if(bs.id == node_id):
                return bs

        for uav in self.uavs:
            if(uav.id == node_id):
                return uav
        return None

    """
    non-LoS links are not considered in the interference.
    """

    def check_user_requests(self):
        count_new_connections = 0
        for user in self.users:
            if(user.request_to_connect):
                [node, best_snr, found_node] = self.assign_user(user)
                if(found_node):
                    user.request_to_connect = False
                    user.connected = True
                    user.connected_to = node.id
                    user.service_provider = node
                    count_new_connections += 1
        return count_new_connections
    
    def check_instant_rate(self):
        for user in self.users:
            user.check_instant_rate()

    # def update_user_condition(self):
    #     for bs in self.base_stations:
    #         for user in bs.users:
    #             interference = self.interference(user)
    #             user.bit_rate = instant_rate(bs, user, interference, self.f_c, self.eta, h_tilde=1)
    #             user.check_instant_rate()

    #     for uav in self.uavs:
    #         for user in uav.users:
    #             interference = self.interference(user)
    #             user.bit_rate = instant_rate(bs, user, interference, self.f_c, self.eta, h_tilde=1)
    #             user.check_instant_rate()

    def interference(self, user):
        interference_bs = []
        interference_uav = []
        for bs in self.base_stations:
            if(LoS(node_1_location=user.location, node_2_location=bs.location, buildings=self.buildings)):
                interference_bs.append(bs.power * channel_gain(user.location, bs.location, self.f_c, self.eta))
        for uav in self.uavs:
            if(LoS(node_1_location=user.location, node_2_location=uav.location, buildings=self.buildings)):
                interference_uav.append(uav.power * channel_gain(user.location, uav.location, self.f_c, self.eta))
        return np.sum(interference_bs) + np.sum(interference_uav) + self.sigma

    def get_state(self):
        return [[i.get_location for i in self.base_stations], [i.get_location for i in self.uavs], [i.get_location for i in self.users]]

    def assign_user(self, user):
        best_snr = 0
        service_provider = None
        # log.info('%7s requested to connect...' % (user.id))
        for bs in self.base_stations:
            if(LoS(node_1_location=user.location, node_2_location=bs.location, buildings=self.buildings)):
                snr = bs.power * channel_gain(user.location, bs.location, self.f_c, self.eta)
                if(snr > best_snr):
                    best_snr = snr
                    service_provider = bs
                    

        for uav in self.uavs:
            if(LoS(node_1_location=user.location, node_2_location=uav.location, buildings=self.buildings)):
                snr = uav.power * channel_gain(user.location, uav.location, self.f_c, self.eta)
                if(snr > best_snr):
                    best_snr = snr
                    service_provider = uav

        if(service_provider == None):
            # log.info('There is no link for %s' %(user.id))
            user.connected = False
            return [None, 0, False]

        if(service_provider.append_user(user)):
            user.connected = True
            # log.info('%7s has been connected to %s with snr %s' %(user.id, service_provider.id, best_snr))
            return [service_provider, best_snr, True]

        # log.info('%7s attempt tp connect to the network failed' %(user.id))
        return [None, 0, False]

    def disconnect_user(self, node:SP_Node, user:User):
        if not(node == None):
            node.remove_user(user)
            user.request_to_connect = True
            user.connected = False
            user.connected_to = 'None'
            user.bit_rate = 0

    # def update_link_condition(self):
    #     for bs in self.base_stations:
    #         for user in bs.users:
    #             if(LoS(node_1_location=user.location, node_2_location=bs.location, buildings=self.buildings)):
    #                 interference = self.interference(user)
    #                 user.bit_rate = instant_rate(bs, user, interference, self.f_c, self.eta, h_tilde=1)
    #                 user.bit_rate = overall_rate(self.uavs, self.base_stations, user, self.buildings, interference, self.f_c, self.eta)
    #                 user.check_instant_rate()
    #             else:
    #                 self.disconnect_user(bs, user)


    #     for uav in self.uavs:
    #         for user in uav.users:
    #             if(LoS(node_1_location=user.location, node_2_location=uav.location, buildings=self.buildings)):
    #                 interference = self.interference(user)
    #                 user.bit_rate = instant_rate(uav, user, interference, self.f_c, self.eta, h_tilde=1)
    #                 user.check_instant_rate()
    #             else:
    #                 self.disconnect_user(uav, user)
    
    def cal_user_bit_rate(self, user):
        if(not user.service_provider or not user.connected):
            user.bit_rate = 0
        else:
            noise = self.interference(user)
            user.bit_rate = overall_rate(self.uavs, self.base_stations, user, self.buildings, noise, self.f_c, self.eta)
            # user.bit_rate = instant_rate(user.service_provider, user, interference, self.f_c, self.eta, h_tilde=1)
            if(user.bit_rate == 0):
                self.disconnect_user(user.service_provider, user)
            user.check_instant_rate()
    
    def update_link_condition(self):
        for user in self.users:
            if(not user.service_provider or not user.connected):
                user.bit_rate = 0
            else:
                noise = self.interference(user)
                user.bit_rate = overall_rate(self.uavs, self.base_stations, user, self.buildings, noise, self.f_c, self.eta)
                # user.bit_rate = instant_rate(user.service_provider, user, interference, self.f_c, self.eta, h_tilde=1)
                if(user.bit_rate == 0):
                    self.disconnect_user(user.service_provider, user)
            user.check_instant_rate()

    
    def find_better_node(self, user:User):
        new_snr = 0
        current_snr = user.current_snr
        new_service_provider = None

        for bs in self.base_stations:
            if(LoS(node_1_location=user.location, node_2_location=bs.location, buildings=self.buildings)):
                snr = bs.power * channel_gain(user.location, bs.location, self.f_c, self.eta)
                if(snr > current_snr):
                    new_snr = snr
                    new_service_provider = bs

        for uav in self.uavs:
            if(LoS(node_1_location=user.location, node_2_location=uav.location, buildings=self.buildings)):
                snr = uav.power * channel_gain(user.location, uav.location, self.f_c, self.eta)
                if(snr > current_snr):
                    new_snr = snr
                    new_service_provider = uav

        if(new_service_provider == None):
            return [None, 0, False]
        else:
            return [new_service_provider, new_snr, True]

            
    def hand_off_update(self):
        count_hand_off = 0
        
        for bs in self.base_stations:
            for user in bs.users:
                user.current_snr = bs.power * channel_gain(user.location, bs.location, self.f_c, self.eta)
                [new_node, new_snr, found_node] = self.find_better_node(user)
                if(found_node):
                    self.disconnect_user(bs, user)
                    new_node.append_user(user)
                    user.request_to_connect = False
                    user.connected = True
                    user.connected_to = new_node.id
                    user.current_snr = new_snr
                    # log.info('%7s has been handed off from %7s to %7s' %(user.id, bs.id, new_node.id))
                    count_hand_off += 1
                
        for uav in self.base_stations:
            for user in uav.users:
                user.current_snr = uav.power * channel_gain(user.location, uav.location, self.f_c, self.eta)
                [new_node, new_snr, found_node] = self.find_better_node(user)
                if(found_node):
                    self.disconnect_user(uav, user)
                    new_node.append_user(user)
                    user.request_to_connect = False
                    user.connected = True
                    user.connected_to = new_node.id
                    user.current_snr = new_snr
                    # log.info('%7s has been handed off from %7s to %7s' %(user.id, uav.id, new_node.id))
                    count_hand_off += 1
        
        return count_hand_off

