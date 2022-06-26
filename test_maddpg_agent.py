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
num_users = 50
num_BSs = 2
num_uavs = 2
reward_weights = np.array([1, 1, 1, 1, 0, 0]) / num_users
env = Environment('Env-1', n_users=num_users, n_uavs=num_uavs, n_BSs=num_BSs, flight_time=200, max_user_in_obs=0, reward_weights=reward_weights)

n_agents = env.num_uavs
actor_dims = []
for i in range(n_agents):
    actor_dims.append(env.observation_space.shape[0])
critic_dims = sum(actor_dims)

n_actions = 2
scenario = 'simple'
save_dir = Path('results') / datetime.datetime.now().strftime('%Y-%m-%dT%H-%M-%S')
save_dir.mkdir(parents=True)
save_dir_render = save_dir / 'render_trajectory'
save_dir_render.mkdir(parents=True)

maddpg_agents = MADDPG(actor_dims, critic_dims, n_agents, n_actions, 
                           fc1=16, fc2=32, fc3=64, fc4=128, fc5=256,
                           alpha=1e-1, beta=1e-1, scenario=scenario,
                           chkpt_dir=str(save_dir) + '/tmp/maddpg/')

memory = MultiAgentReplayBuffer(100000, critic_dims, actor_dims, 
                        n_actions, n_agents, batch_size=1024)

logger = MetricLogger(save_dir)
episodes = int(1.5e2)
# logger.record_initials(len(agent.memory), agent.batch_size, agent.exploration_rate_decay, agent.burnin, agent.learn_every, agent.sync_every)
evaluate = False
if evaluate:
    maddpg_agents.load_checkpoint()

score_history = []
best_score = 0

for agent in maddpg_agents.agents:
    agent.noise_type = "param"
    agent.desired_distance = 0.7
    agent.scalar_decay = 0.9999
    agent.scalar = 0.05
    agent.normal_scalar = 0.25

with open(str(save_dir) + '/hyperparams.txt', 'w') as f:
    f.write(str(maddpg_agents.agents[0]))

for e in range(episodes):
    obs = env.reset()
    # for agent in maddpg_agents.agents:
    #     agent.noise.reset()
    done = [False] * n_agents
    score = 0
    while not any(done):
        actions = maddpg_agents.choose_action(obs)
        obs_, reward, done, info = env.step(actions)
        state = obs_list_to_state_vector(obs)
        state_ = obs_list_to_state_vector(obs_)
        memory.store_transition(obs, state, actions, reward, obs_, state_, done)
        if maddpg_agents.curr_step % 16 == 0:
            maddpg_agents.learn(memory)
        logger.log_step(sum(reward), info['num connected users'], info['avg bit rate'])
        obs = obs_
        score += sum(reward)
    score_history.append(score)
    avg_score = np.mean(score_history[-10:])
    if not evaluate:
        if avg_score > best_score:
            maddpg_agents.save_checkpoint()
            best_score = avg_score
    logger.log_episode()
    
    if e % 1 == 0:
        env.render(e, save_dir_render,"trajectory")
        logger.record(
            episode=e,
            epsilon=np.mean(maddpg_agents.agents[0].distances[-200:-1]),
            step=maddpg_agents.curr_step
        )