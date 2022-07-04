import numpy as np
import datetime
from pathlib import Path
from metrics import MetricLogger
from Env.environment import Environment
from Agents.maddpg import MADDPG
from Agents.buffer import MultiAgentReplayBuffer

def obs_list_to_state_vector(observation):
    state = np.array([])
    for obs in observation:
        state = np.concatenate([state, obs])
    return state

# reward = [self.total_bit_rate, uav_total_bit_rate, self.connected_users, len(self.uavs[agent_idx].users), collision_reward, step_reward]
num_users = 100
num_BSs = 0
num_uavs = 3
reward_weights = np.array([10, 10, 0, 0, 1, 0]) / num_users
env = Environment('Env-1', n_users=num_users, n_uavs=num_uavs, n_BSs=num_BSs, users_zones=[1, 3, 4], users_var=30,flight_time=200, max_user_in_obs=0, reward_weights=reward_weights)

n_agents = env.num_uavs
actor_dims = []
for i in range(n_agents):
    actor_dims.append(env.observation_space.shape[0])
critic_dims = sum(actor_dims)

n_actions = 1
scenario = 'simple'
save_dir = Path('results') / datetime.datetime.now().strftime('%Y-%m-%dT%H-%M-%S')
save_dir.mkdir(parents=True)
save_dir_render = save_dir / 'render_trajectory'
save_dir_render.mkdir(parents=True)

logger = MetricLogger(save_dir)
episodes = int(1e2)

score_history = []
best_score = 0

for e in range(episodes):
    obs = env.reset()        
    done = [False] * n_agents
    score = 0
    actions = np.pi * np.ones((n_agents, 1), dtype=np.float64) + np.pi / 2
    while not any(done):
        actions += np.pi / 80
        for action in actions:
            if action > np.pi:
                action -= 2 * np.pi
        actions = np.clip(actions, -np.pi, np.pi)
        obs_, reward, done, info = env.step(actions)
        logger.log_step(np.mean(reward), info['num connected users'], info['avg bit rate'])
        obs = obs_
        score += np.mean(reward)
    score_history.append(score)
    avg_score = np.mean(score_history[-10:])
    logger.log_episode()
    
    if e % 1 == 0:
        env.render(e, save_dir_render,"trajectory")
        logger.record(
            episode=e,
            epsilon=1,
            step=1
        )