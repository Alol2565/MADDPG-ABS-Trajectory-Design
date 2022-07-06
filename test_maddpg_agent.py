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
num_users = 10
num_BSs = 0
num_uavs = 1
users_zones=[1,2,3,4]
users_var=40
max_user_in_obs=10
normal_scalar_decay = 0.8
learning_period = 16
reward_weights = np.array([10, 10, 0, 0, 1, 0]) / num_users
env = Environment('Env-1', n_users=num_users, n_uavs=num_uavs, n_BSs=num_BSs, users_zones=users_zones,users_var=users_var ,flight_time=200, max_user_in_obs=max_user_in_obs, reward_weights=reward_weights)

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

maddpg_agents = MADDPG(actor_dims, critic_dims, n_agents, n_actions, 
                           fc1=400, fc2=300, fc3=200, fc4=200, fc5=256,
                           alpha=3e-1, beta=1e-2, scenario=scenario,
                           chkpt_dir=str(save_dir) + '/tmp/maddpg/')

memory = MultiAgentReplayBuffer(100000, critic_dims, actor_dims, 
                        n_actions, n_agents, batch_size=1024)

logger = MetricLogger(save_dir)
episodes = int(1e3)
# logger.record_initials(len(agent.memory), agent.batch_size, agent.exploration_rate_decay, agent.burnin, agent.learn_every, agent.sync_every)
evaluate = False
if evaluate:
    maddpg_agents.load_checkpoint()

score_history = []
best_score = 0

for agent in maddpg_agents.agents:
    agent.noise_type = "normal"
    agent.normal_scalar = np.pi / 2
    agent.normal_scalar_decay = 1

stop_learning = False
learning_rate_decay = True

with open(str(save_dir) + "/specs.txt", "w") as text_file:
    text_file.write("Number of Users: %s\n" %(num_users))
    text_file.write("Number of Base Stations: %s\n" %(num_BSs))
    text_file.write("Number of UAVs: %s\n" %(num_uavs))
    text_file.write("Reward Weights: %s\n" %(reward_weights))
    text_file.write("Maximum user in observation: %s\n" %max_user_in_obs)
    text_file.write("Variance of Users: %s\n" %users_var)
    text_file.write("Users Zones: %s\n" %users_zones)
    text_file.write("Normal Scalar: %s\n" %maddpg_agents.agents[0].normal_scalar)
    text_file.write("Normal Scalar Decay: %s\n" %(normal_scalar_decay))
    text_file.write("Learning Period: %s\n" %(learning_period))


for e in range(episodes):
    obs = env.reset()
    if(maddpg_agents.agents[0].normal_scalar < 0.25):
        stop_learning = True
        for agent in maddpg_agents.agents:
            agent.normal_scalar_decay = 1
            agent.normal_scalar = 0.25
    else:
        if e % 10 == 0:
            for agent in maddpg_agents.agents:
                agent.normal_scalar *= normal_scalar_decay
        
    done = [False] * n_agents
    score = 0
    while not any(done):
        actions = maddpg_agents.choose_action(obs)
        obs_, reward, done, info = env.step(actions)
        state = obs_list_to_state_vector(obs)
        state_ = obs_list_to_state_vector(obs_)
        memory.store_transition(obs, state, actions, reward, obs_, state_, done)
        if maddpg_agents.curr_step % learning_period == 0 and not stop_learning:
            maddpg_agents.learn(memory)
        logger.log_step(np.mean(reward), info['num connected users'], info['avg bit rate'])
        obs = obs_
        score += np.mean(reward)
    print('lr actor: {0} lr critic: {1}'.format(maddpg_agents.agents[0].actor.optimizer.param_groups[0]['lr'], 
                maddpg_agents.agents[0].critic.optimizer.param_groups[0]['lr']))
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
            epsilon=maddpg_agents.agents[0].normal_scalar,
            step=maddpg_agents.curr_step
        )