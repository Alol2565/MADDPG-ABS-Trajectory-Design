import numpy as np
import datetime
from pathlib import Path
from sympy import false
from metrics import MetricLogger
from Env.environment import Environment
import torch
from Agents.maddpg import MADDPG
from Agents.buffer import MultiAgentReplayBuffer

def obs_list_to_state_vector(observation):
    state = np.array([])
    for obs in observation:
        state = np.concatenate([state, obs])
    return state

env = Environment('Env-1', n_users=30, n_uavs=2, n_BSs=0, flight_time=200, max_user_in_obs=0)

n_agents = env.num_uavs
actor_dims = []
for i in range(n_agents):
    actor_dims.append(env.observation_space.shape[0])
critic_dims = sum(actor_dims)
# action space is a list of arrays, assume each agent has same action space
n_actions = 2
scenario = 'simple'


save_dir = Path('results') / datetime.datetime.now().strftime('%Y-%m-%dT%H-%M-%S')
save_dir.mkdir(parents=True)
save_dir_render = save_dir / 'render_trajectory'
save_dir_render.mkdir(parents=True)

maddpg_agents = MADDPG(actor_dims, critic_dims, n_agents, n_actions, 
                           fc1=256, fc2=512, fc3=1024, fc4=2048, fc5=4096,
                           alpha=0.01, beta=0.01, scenario=scenario,
                           chkpt_dir=str(save_dir) + '/tmp/maddpg/')

memory = MultiAgentReplayBuffer(100000, critic_dims, actor_dims, 
                        n_actions, n_agents, batch_size=1024)

logger = MetricLogger(save_dir)
episodes = int(1e2)
# logger.record_initials(len(agent.memory), agent.batch_size, agent.exploration_rate_decay, agent.burnin, agent.learn_every, agent.sync_every)
evaluate = False

if evaluate:
    maddpg_agents.load_checkpoint()

score_history = []
best_score = 0
for e in range(episodes):
    obs = env.reset()
    for agent in maddpg_agents.agents:
        # agent.noise.sigma *= 1.0 / (1.0 + e / 100.0)
        agent.noise.sigma = 1.0
    done = [False] * n_agents
    score = 0
    while not any(done):
        actions = maddpg_agents.choose_action(obs)
        obs_, reward, done, info = env.step(actions)
        state = obs_list_to_state_vector(obs)
        state_ = obs_list_to_state_vector(obs_)

        memory.store_transition(obs, state, actions, reward, obs_, state_, done)

        if maddpg_agents.curr_step % 32 == 0 and not evaluate:
            maddpg_agents.learn(memory)

        logger.log_step(sum(reward), 0, 0)

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
        mean_bit_rate = np.mean(env.bit_rate_each_ep[-500:])
        logger.record(
            episode=e,
            epsilon=maddpg_agents.agents[0].noise.sigma,
            step=maddpg_agents.curr_step,
            mean_bit_rate=mean_bit_rate
        )