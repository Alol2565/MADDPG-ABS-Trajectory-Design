from typing import List
import matplotlib.pyplot as plt
from Env.network import Network
from Env.BaseStations.base_station import Base_Station
from Env.Users.user import User
from Env.building import Building
from Env.BaseStations.uav import UAV
import datetime
import numpy as np
import concurrent.futures


class Environment:
    def __init__(self, id, n_users=10, n_uavs=1, n_BSs=0, users_zones=[3], users_var=30,flight_time=3600, max_user_in_obs=5, reward_weights=np.array([1, 1, 1, 1, 1])):
        self.id = id
        self.time_res = 1
        self.max_user_uav = max_user_in_obs
        self.multi_agent = n_uavs > 1
        self.flight_time = flight_time
        self.reward_weights = reward_weights
        self.users_zones = users_zones
        self.users_var = users_var
        """
        Space Borders
        """
        self.MAX_X = 400
        self.MIN_X = 0
        self.MAX_Y = 0
        self.MIN_Y = -380
        self.MAX_Z = 100
        self.MIN_Z = -10
        self.borders = [np.array([self.MIN_X, self.MIN_Y, self.MIN_Z]), np.array([self.MAX_X, self.MAX_Y, self.MAX_Z])]
        self.time = 0
        """
        Buildings definition
        """
        self.buildings:List[Building] = []
        building_id = 'building_1'
        location = np.array([29, -33.3, 0], dtype=np.float64)
        self.buildings.append(Building(building_id, location, length=37.5, width=40, height=25))

        building_id = 'building_2'
        location = np.array([36, -102, 0], dtype=np.float64)
        self.buildings.append(Building(building_id, location, length=44, width=79, height=15))

        building_id = 'building_3'
        location = np.array([36, -219, 0], dtype=np.float64)
        self.buildings.append(Building(building_id, location, length=46, width=108, height=15))

        building_id = 'building_4'
        location = np.array([111, -93, 0], dtype=np.float64)
        self.buildings.append(Building(building_id, location, length=49, width=60, height=10))

        building_id = 'building_5'
        location = np.array([105, -172, 0], dtype=np.float64)
        self.buildings.append(Building(building_id, location, length=57, width=20, height=12))

        building_id = 'building_6'
        location = np.array([96, -262, 0], dtype=np.float64)
        self.buildings.append(Building(building_id, location, length=44, width=51, height=7))

        building_id = 'building_7'
        location = np.array([162, -120, 0], dtype=np.float64)
        self.buildings.append(Building(building_id, location, length=40, width=40, height=15))

        building_id = 'building_8'
        location = np.array([158, -180, 0], dtype=np.float64)
        self.buildings.append(Building(building_id, location, length=17, width=32, height=15))

        building_id = 'building_9'
        location = np.array([161, -215, 0], dtype=np.float64)
        self.buildings.append(Building(building_id, location, length=16, width=32, height=9))

        building_id = 'building_10'
        location = np.array([202, -235, 0], dtype=np.float64)
        self.buildings.append(Building(building_id, location, length=40, width=38, height=10))

        building_id = 'building_11'
        location = np.array([205, -177, 0], dtype=np.float64)
        self.buildings.append(Building(building_id, location, length=63, width=17, height=15))

        building_id = 'building_12'
        location = np.array([286, -220, 0], dtype=np.float64)
        self.buildings.append(Building(building_id, location, length=35, width=33, height=42))

        building_id = 'building_13'
        location = np.array([232, -312, 0], dtype=np.float64)
        self.buildings.append(Building(building_id, location, length=32, width=32, height=12))

        building_id = 'building_14'
        location = np.array([286, -303, 0], dtype=np.float64)
        self.buildings.append(Building(building_id, location, length=35, width=33, height=42))

        building_id = 'building_15'
        location = np.array([340, -307, 0], dtype=np.float64)
        self.buildings.append(Building(building_id, location, length=38, width=69, height=15))

        building_id = 'building_16'
        location = np.array([340, -207, 0], dtype=np.float64)
        self.buildings.append(Building(building_id, location, length=36, width=57, height=15))

        building_id = 'building_17'
        location = np.array([225, -82, 0], dtype=np.float64)
        self.buildings.append(Building(building_id, location, length=50, width=34, height=30))

        """
        Users definition
        """
        self.num_users = n_users
        self.connected_users = 0
        self.users:List[User] = self.num_users * [None]
        users_id = ['user_' + str(i) for i in range(self.num_users)]
        """
        Base Stations definition
        """
        self.num_BSs = n_BSs
        self.BSs: List[Base_Station] = self.num_BSs * [None]
        BSs_id = ['BS_' + str(i) for i in range(self.num_BSs)]
        BSs_location = [np.array([390, -10, 20], dtype=np.float64), np.array([390, -370, 20], dtype=np.float64)] 

        """
        UAVs definition
        """
        self.uav_obs_range = 10000

        self.num_uavs = n_uavs
        self.uavs:List[UAV] = self.num_uavs * [None]
        uavs_id = ['uav_' + str(i) for i in range(self.num_uavs)]
        uavs_location = self.num_uavs * [np.array([50., -50., 50.], dtype=np.float64)]

        with concurrent.futures.ThreadPoolExecutor() as executor:
            gen_ob_user = executor.map(self.create_user, users_id)
            gen_ob_bs = executor.map(self.create_base_station, BSs_id, BSs_location)
            gen_ob_uav = executor.map(self.create_uav, uavs_id, uavs_location)
            self.users = list(gen_ob_user)
            self.BSs = list(gen_ob_bs)
            self.uavs = list(gen_ob_uav)

        """
        Initializing Network Topology
        """
        self.network = Network(users=self.users,
                                   BSs=self.BSs,
                                   uavs=self.uavs,
                                   buildings=self.buildings,
                                   f_c=5e6,
                                   eta=1e-2,
                                   sigma=1e-3)
        self.total_bit_rate = 0
        self.bit_rate_each_ep = []

        self.frames = np.ndarray(shape=(self.num_uavs, self.max_user_uav + 1, 3), dtype=np.float64)
        self.observation_space = np.array(self.frames[0,:]).flatten()

    # def create_user(self, user_id):
    #     user_location = np.array([
    #     np.random.uniform(self.borders[0][0], self.borders[1][0]),
    #     np.random.uniform(self.borders[0][1], self.borders[1][1]),
    #     1
    #     ])
    #     return User(id=user_id, power=1, initial_location=user_location, env_borders=self.borders)

    def create_user(self, user_id):
        zone = np.random.choice(self.users_zones, 1)
        # zone = 0
        if(zone == 1):
            zone_x = 1 / 4
            zone_y = 1 / 4
        if(zone == 2):
            zone_x = 3 / 4
            zone_y = 1 / 4
        if(zone == 3):
            zone_x = 1 / 4
            zone_y = 3 / 4
        if(zone == 4):
            zone_x = 3 / 4
            zone_y = 3 / 4 

        user_location = np.array([
        np.random.normal((self.borders[1][0] - self.borders[0][0]) * zone_x, self.users_var),
        np.random.normal((self.borders[0][1] - self.borders[1][1]) * zone_y, self.users_var), 1])

        for i in range(3):
            if user_location[i] < self.borders[0][i]:
                user_location[i] = self.borders[0][i] / 2
            elif user_location[i] > self.borders[1][i]:
                user_location[i] = self.borders[1][i] / 2
        return User(id=user_id, power=1, initial_location=user_location, env_borders=self.borders)

    def create_base_station(self, BS_id, location):
        return Base_Station(id=BS_id, power=1, initial_location=location)

    def create_uav(self, uav_id, location):
        return UAV(id=uav_id, power=1e-2, initial_location=location, env_borders=self.borders, buildings=self.buildings)

    def observe(self, agent_idx):
        distances = []
        for user in self.users:
            distances.append(np.linalg.norm(user.location - self.uavs[agent_idx].location))
        sorted_distances = np.argsort(distances)
        close_users_idx = sorted_distances[:self.max_user_uav]
        for i in range(self.max_user_uav):
            self.frames[agent_idx][i] = self.users[close_users_idx[i]].location - self.uavs[agent_idx].location
        self.frames[agent_idx][self.max_user_uav] = self.uavs[agent_idx].location
        # self.frames[agent_idx][self.max_user_uav + 1] = self.uavs[agent_idx].velocity
        return self.frames[agent_idx].flatten()
        
    def in_no_fly_zone(self, agent_location):
        """
        Some areas should be considered as no fly zone like airports. also buildings' volume should be assessd as no fly zone area 
        to avoid collision of uavs with buildings.
        """
        return False

    def lost_connectivity(self, agent_location):
        """
        UAVs should maintain their connectiviy with BSs. some routes between uavs can be identified as path to connect to BSs. Now for simplicity,
        we consider that all uavs are connected. 
        """
        return False

    def reward_function(self, agent_idx):
        reward_area = 0
        reward_connectivity = 0
        reward_bit_rate = 0
        self.total_bit_rate = 0
        collision_reward = 0
        uav_total_bit_rate = 0
        self.connected_users = 0

        for user in self.users:
            self.total_bit_rate += user.bit_rate
            if user.bit_rate > 0:
                self.connected_users += 1

        self.avg_bit_rate = self.total_bit_rate / self.num_users
        
        if(self.in_no_fly_zone(self.uavs[agent_idx].location)):
            reward_area = -1

        if(self.lost_connectivity(self.uavs[agent_idx].location)):
            reward_connectivity = -1

        if(self.total_bit_rate < 0.01 * len(self.users)):
            reward_bit_rate = -1
        
        if(self.uavs[agent_idx].collision):
            collision_reward = -1
        
        for user in self.uavs[agent_idx].users:
            uav_total_bit_rate += user.bit_rate
        step_reward = 0.0
        reward = [self.total_bit_rate, uav_total_bit_rate, self.connected_users, len(self.uavs[agent_idx].users), collision_reward, step_reward]
        return np.matmul(self.reward_weights, reward)

    def move_user(self, user, delta_time):
        return user.move(delta_time)

    def move_uav(self, uav, action, delta_time):
        return uav.move(action, delta_time)

    def step(self, actions):
        delta_time = np.float64(1)
        self.time += delta_time
        done = self.num_uavs * [False]
        if(self.time >= self.flight_time):
            done = self.num_uavs * [True]

        for uav_idx in range(self.num_uavs):
            if(self.uavs[uav_idx].out_of_battery):
                done[uav_idx] = True

        # if self.multi_agent:

        with concurrent.futures.ThreadPoolExecutor() as executor:
            executor.map(self.move_user, self.users, self.num_users * [delta_time])
            executor.map(self.move_uav, self.uavs, actions, self.num_uavs * [delta_time])

        # else:
        #     with concurrent.futures.ThreadPoolExecutor() as executor:
        #         executor.map(self.move_user, self.users, self.num_users * [delta_time])
        #         executor.map(self.move_uav, self.uavs, self.num_uavs * [actions], self.num_uavs * [delta_time])

            
        num_new_connections = self.network.check_user_requests()
        # log.info('%3s new connections has been established' %(num_new_connections))
        self.network.update_link_condition()
        num_hand_off = self.network.hand_off_update()
        # log.info('%5s hand-off has happened' %(num_hand_off))

        # obs = np.zeros((self.num_uavs, len(self.observation_space[0])))
        obs = np.ndarray(shape=(self.num_uavs, len(self.observation_space)))
        rewards = []
        for uav_idx in range(self.num_uavs):
            obs[uav_idx, :] = self.observe(uav_idx)
            rewards.append(self.reward_function(uav_idx))
        obs = np.array(obs, dtype=np.float32)
        rewards = np.array(rewards, dtype=np.float32)
        info = {'avg bit rate': self.avg_bit_rate, 'num connected users': self.connected_users}
        # log_ml.info('Time: %.2f |  reward: %.2f  |  observation: %s  |  action: %s  |  done: %s' %(self.time, reward, obs, action, done))
        self.bit_rate_each_ep.append(self.total_bit_rate)
        return [obs, rewards, done, info]


    def reset_building(self, building_idx):
        self.buildings[building_idx].reset()

    def reset_bs(self, bs_idx):
        self.BSs[bs_idx].reset()

    def reset_uav(self, uav_idx):
        self.uavs[uav_idx].reset()

    def reset_user(self, user_idx):
        self.users[user_idx].reset()

    def reset(self):
        self.time = 0
        for building in self.buildings:
            building.reset()

        for uav in self.uavs:
            uav.reset()

        for bs in self.BSs:
            bs.reset()

        for user in self.users:
            user.reset()

        self.bit_rate_each_ep = []

        # log.info('All objects have been reset')

        obs = []
        for uav_idx in range(self.num_uavs):
            obs.append(self.observe(uav_idx))
        obs = np.array(obs, dtype=np.float32)
        return obs

        
        
        """Resets the environment to an initial state and returns an initial
        observation.

        This method should also reset the environment's random number
        generator(s) if `seed` is an integer or if the environment has not
        yet initialized a random number generator. If the environment already
        has a random number generator and `reset` is called with `seed=None`,
        the RNG should not be reset.
        Moreover, `reset` should (in the typical use case) be called with an
        integer seed right after initialization and then never again.

        Returns:
            observation (object): the initial observation.
            info (optional dictionary): a dictionary containing extra information, this is only returned if return_info is set to true
        """

    def render(self, iter, save_dir,mode="frame"):
        if(mode=="frame"):
            for building in self.buildings:
                pts = building.points()
                rectangle = plt.Rectangle(pts[0], pts[1], pts[2],color='#E2E2E2', alpha=1)
                plt.gca().add_patch(rectangle)
                plt.text(building.location[0],building.location[1], building.id,
                    horizontalalignment='center', verticalalignment='center',
                    fontsize=7, color='#9B9B9B')


            for uav in self.uavs:
                pt_x = uav.location[0]
                pt_y = uav.location[1]
                plt.scatter(pt_x, pt_y, c='blue', marker ="^", edgecolor ="red", cmap=uav.id, linewidths=0.1, alpha=1)
                plt.text(uav.location[0],uav.location[1] - 3, uav.id,
                    horizontalalignment='center', verticalalignment='center',
                    fontsize=7, color='blue')

            for user in self.users:
                pt_x = user.location[0]
                pt_y = user.location[1]
                user_color = 'red'
                if(user.connected):
                    user_color = 'green'

                plt.scatter(pt_x, pt_y, c=user_color, marker =".", linewidths=0.05, alpha=1)
                plt.text(user.location[0],user.location[1] - 5, user.id +' | ' + user.connected_to,
                    horizontalalignment='center', verticalalignment='center',
                    fontsize=6, color=user_color)


            for bs in self.BSs:
                pt_x = bs.location[0]
                pt_y = bs.location[1]
                plt.scatter(pt_x, pt_y, c='black', marker ="^", linewidths=4, alpha=1)
                plt.text(bs.location[0],bs.location[1] - 7, bs.id,
                    horizontalalignment='center', verticalalignment='center',
                    fontsize=7, color='black')
                

            # plt.axis('scaled')
            plt.xlim(self.MIN_X, self.MAX_X)
            plt.ylim(self.MIN_Y, self.MAX_Y)
            figure = plt.gcf()
            figure.set_size_inches(16, 9.12)
            plt.tight_layout()
            plt.savefig('render_results/fig_' + str(iter) + '_' + datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S'), dpi=100)
            plt.clf()
            return 

        if(mode=="trajectory"):
            for building in self.buildings:
                pts = building.points()
                rectangle = plt.Rectangle(pts[0], pts[1], pts[2],color='#E2E2E2', alpha=1)
                plt.gca().add_patch(rectangle)
                plt.text(building.location[0],building.location[1], building.id,
                    horizontalalignment='center', verticalalignment='center',
                    fontsize=7, color='#9B9B9B')

            for i in range(self.num_uavs):
                self.uavs[i].trajectory = np.array(self.uavs[i].trajectory)
                plt.plot(self.uavs[i].trajectory[:,0], self.uavs[i].trajectory[:,1], (0.1, i / self.num_uavs, 0.5))
                # plt.text(uav.location[0],uav.location[1] - 3, uav.id,
                #     horizontalalignment='center', verticalalignment='center',
                #     fontsize=7, color='blue')

            for user in self.users:
                pt_x = user.location[0]
                pt_y = user.location[1]
                user_color = 'red'
                if(user.connected):
                    user_color = 'green'
                plt.scatter(pt_x, pt_y, c=user_color, marker =".", linewidths=0.05, alpha=1)
                plt.text(user.location[0],user.location[1] - 5, user.id +' | ' + user.connected_to + ' | ' + str(user.mean_bit_rate),
                    horizontalalignment='center', verticalalignment='center',
                    fontsize=6, color=user_color)

            for bs in self.BSs:
                pt_x = bs.location[0]
                pt_y = bs.location[1]
                plt.scatter(pt_x, pt_y, c='black', marker ="^", linewidths=4, alpha=1)
                plt.text(bs.location[0],bs.location[1] - 7, bs.id,
                    horizontalalignment='center', verticalalignment='center',
                    fontsize=7, color='black')

            plt.xlim(self.MIN_X, self.MAX_X)
            plt.ylim(self.MIN_Y, self.MAX_Y)
            figure = plt.gcf()
            figure.set_size_inches(16, 9.12)
            plt.tight_layout()
            plt.savefig(str(save_dir) + '/fig_' + str(iter), dpi=100)
            plt.clf()
            return 

        if(mode=="3d"):
            fig = plt.figure()
            ax = plt.axes(projection='3d')
            # for building in self.buildings:
            #     pts = building.points()
            #     rectangle = plt.Rectangle(pts[0], pts[1], pts[2],color='#E2E2E2', alpha=1)
            #     ax.plot3D.gca().add_patch(rectangle)
            #     plt.text(building.location[0],building.location[1], building.id,
            #         horizontalalignment='center', verticalalignment='center',
            #         fontsize=7, color='#9B9B9B')

            for i in range(self.num_uavs):
                self.uavs[i].trajectory = np.array(self.uavs[i].trajectory)
                ax.plot3D(self.uavs[i].trajectory[:,0], self.uavs[i].trajectory[:,1], self.uavs[i].trajectory[:,2], (0.1, i / self.num_uavs, 0.5))
                # plt.text(uav.location[0],uav.location[1] - 3, uav.id,
                #     horizontalalignment='center', verticalalignment='center',
                #     fontsize=7, color='blue')

            for user in self.users:
                pt_x = user.location[0]
                pt_y = user.location[1]
                pt_z = user.location[2]
                user_color = 'red'
                if(user.connected):
                    user_color = 'green'
                ax.scatter3D(pt_x, pt_y, pt_z, c=user_color, marker =".", linewidths=0.05, alpha=1)
                ax.text(user.location[0],user.location[1] - 5, user.id +' | ' + user.connected_to + ' | ' + str(user.mean_bit_rate),
                    horizontalalignment='center', verticalalignment='center',
                    fontsize=6, color=user_color)

            for bs in self.BSs:
                pt_x = bs.location[0]
                pt_y = bs.location[1]
                pt_z = bs.location[2]
                ax.scatter3D(pt_x, pt_y, pt_z, c='black', marker ="^", linewidths=4, alpha=1)
                ax.text(bs.location[0],bs.location[1] - 7, bs.id,
                    horizontalalignment='center', verticalalignment='center',
                    fontsize=7, color='black')

            ax.xlim(self.MIN_X, self.MAX_X)
            ax.ylim(self.MIN_Y, self.MAX_Y)
            ax.zlim(self.MIN_Z, self.MAX_Z)
            figure = ax.gcf()
            figure.set_size_inches(16, 9.12)
            ax.tight_layout()
            ax.savefig(str(save_dir) + '/fig_' + str(iter), dpi=100)
            ax.clf()
            return 

        """Renders the environment.

        The set of supported modes varies per environment. (And some
        third-party environments may not support rendering at all.)
        By convention, if mode is:

        - human: render to the current display or terminal and
          return nothing. Usually for human consumption.
        - rgb_array: Return an numpy.ndarray with shape (x, y, 3),
          representing RGB values for an x-by-y pixel image, suitable
          for turning into a video.
        - ansi: Return a string (str) or StringIO.StringIO containing a
          terminal-style text representation. The text can include newlines
          and ANSI escape sequences (e.g. for colors).

        Note:
            Make sure that your class's metadata 'render.modes' key includes
              the list of supported modes. It's recommended to call super()
              in implementations to use the functionality of this method.

        Args:
            mode (str): the mode to render with

        Example:

        class MyEnv(Env):
            metadata = {'render.modes': ['human', 'rgb_array']}

            def render(self, mode='human'):
                if mode == 'rgb_array':
                    return np.array(...) # return RGB frame suitable for video
                elif mode == 'human':
                    ... # pop up a window and render
                else:
                    super(MyEnv, self).render(mode=mode) # just raise an exception
        """