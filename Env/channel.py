from cmath import sin
from platform import node
import numpy as np
import scipy.constants
from Env.BaseStations.base_station import Base_Station
from Env.BaseStations.uav import UAV
from Env.building import Building
from typing import List
from Env.Users.user import User

"""
1. [X] Path Loss
2. [ ] Shadowing
3. [ ] small-scale fading
"""

def LoS(node_1_location, node_2_location, buildings:List[Building], threshold=1):
    number_planes = 0
    for building in buildings:
        number_planes += building.plane_num(node_1_location, node_2_location)
    return number_planes <= threshold

def channel_gain(node_1_location, node_2_location, f_c, eta, h_tilde=1):
    distance = np.linalg.norm(node_1_location - node_2_location)
    rho =  np.power((4 * np.pi * f_c) / scipy.constants.c, 2) * eta
    g = (h_tilde / rho) / np.power(distance, 2)
    return g

def SINR(user, node_t, interference,f_c, eta, h_tilde=1):
    signal_power = node_t.power * channel_gain(user.location, node_t.location, f_c, eta)
    sinr = signal_power / (interference - signal_power)
    return sinr

# def instant_rate(node_r, node_t, interference, f_c, eta, h_tilde=1):
#     # sinr = SINR(node_r, node_t, interference,f_c, eta, h_tilde)
#     # rate = np.log2(1 + (sinr))
#     return rate

def overall_rate(UAVs:List[UAV], BSs:List[Base_Station], user:User, buildings:List[Building],interference, f_c, eta, h_tilde=1):
    overall_rate = 0
    for uav in UAVs:
        if(LoS(node_1_location=user.location, node_2_location=uav.location, buildings=buildings)):
            sinr = SINR(user, uav, interference,f_c, eta, h_tilde)
            if(1 + sinr < 0):
                raise 'Wrong in calculating bit rate'
            overall_rate += np.log2(1 + (sinr))
    for bs in BSs:
        if(LoS(node_1_location=user.location, node_2_location=bs.location, buildings=buildings)):
            sinr = SINR(user, bs, interference,f_c, eta, h_tilde)
            if(1 + sinr < 0):
                raise 'Wrong in calculating bit rate'
            overall_rate += np.log2(1 + (sinr))
    return overall_rate
