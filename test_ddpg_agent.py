import numpy as np
import datetime
from pathlib import Path
from sympy import false
from metrics import MetricLogger
from Env.environment import Environment
import torch
from Agents.ddpg_agent import DDPGAgnet

env = Environment('Env-1', n_users=40, n_uavs=1, n_BSs=0, obs_type='1D')

agent = DDPGAgnet(alpha=0.0001, beta=0.001, 
                input_dims=[64], tau=0.001,
                batch_size=64, fc1_dims=400, fc2_dims=300, 
                n_actions=2)

save_dir = Path('results') / datetime.datetime.now().strftime('%Y-%m-%dT%H-%M-%S')
save_dir.mkdir(parents=True)
save_dir_render = save_dir / 'render_trajectory'
save_dir_render.mkdir(parents=True)
save_dir_model = save_dir / 'trained_model'
save_dir_model.mkdir(parents=True)

logger = MetricLogger(save_dir)

episodes = int(1e3)

# logger.record_initials(len(agent.memory), agent.batch_size, agent.exploration_rate_decay, agent.burnin, agent.learn_every, agent.sync_every)

for e in range(episodes):
    state = env.reset()
    done = False
    agent.noise.reset()
    while not done:
        
        action = agent.act(state)
        next_state, reward, done, info = env.step(action)
        agent.cache(state, action, reward, next_state, done)
        loss = agent.learn()
        logger.log_step(reward, loss, 1)
        state = next_state
    logger.log_episode()
    if e % 1 == 0:
        env.render(e, save_dir_render,"trajectory")
        mean_bit_rate = np.mean(env.bit_rate_each_ep[-3600:])
        logger.record(
            episode=e,
            epsilon=0,
            step=agent.curr_step,
            mean_bit_rate=mean_bit_rate
        )


