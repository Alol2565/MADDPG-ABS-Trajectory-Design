import torch as T
from Agents.networks import ActorNetwork, CriticNetwork
import numpy as np
from Agents.noise import OUActionNoise

class Agent:
    def __init__(self, actor_dims, critic_dims, n_actions, n_agents, agent_idx, chkpt_dir,
                    alpha=0.01, beta=0.01, fc1=64, 
                    fc2=64, fc3=128, fc4=256, fc5=512, gamma=0.95, tau=0.01, noise_type="param", desired_distance=0.7, 
                    scalar_decay=0.99, scalar=0.05, normal_scalar=0.25):
        self.gamma = gamma
        self.tau = tau
        self.n_actions = n_actions
        self.agent_name = 'agent_%s' % agent_idx

        
        self.actor = ActorNetwork(alpha, actor_dims, fc1, fc2, fc3, fc4, fc5, n_actions, 
                                  chkpt_dir=chkpt_dir,  name=self.agent_name+'_actor')

        self.actor_noised = ActorNetwork(alpha, actor_dims, fc1, fc2, fc3, fc4, fc5, n_actions, 
                                  chkpt_dir=chkpt_dir,  name=self.agent_name+'_actor_noised')
        self.critic = CriticNetwork(beta, critic_dims, 
                            fc1, fc2, fc3, fc4, fc5,n_agents, n_actions, 
                            chkpt_dir=chkpt_dir, name=self.agent_name+'_critic')
        self.target_actor = ActorNetwork(alpha, actor_dims, fc1, fc2, fc3, fc4, fc5,n_actions,
                                        chkpt_dir=chkpt_dir, 
                                        name=self.agent_name+'_target_actor')
        self.target_critic = CriticNetwork(beta, critic_dims, 
                                            fc1, fc2, fc3, fc4, fc5, n_agents, n_actions,
                                            chkpt_dir=chkpt_dir,
                                            name=self.agent_name+'_target_critic')

        self.update_network_parameters(tau=1)
        self.epsilon = 1
        self.noise_type = noise_type
        self.distances = []
        self.desired_distance = desired_distance
        self.scalar_decay = scalar_decay
        self.scalar = scalar
        self.normal_scalar = normal_scalar
        self.ou_noise = OUActionNoise(size=2, mu=0, sigma=0.2, theta=0.15)
        self.normal_scalar_decay = 0.99995
        self.hyperparameters = '\n'.join(f"{key:>17}: {value}" for key, value in locals().items() if key != 'self')

    # def choose_action(self, observation):
    #     state = T.tensor(observation, dtype=T.float).to(self.actor.device)
    #     actions = self.actor.forward(state)
    #     action = actions + T.tensor(self.noise(), dtype=T.float).to(self.actor.device)
    #     return action.detach().cpu().numpy()

    def choose_action(self, observation, add_noise=True):
        """Returns actions for given state as per current policy."""
        state = T.tensor(observation, dtype=T.float).to(self.actor.device)
        actions = np.pi * self.actor.forward(state).cpu().data.numpy()
        if add_noise:
            if self.noise_type == "param":
                # hard copy the actor_regular to actor_noised
                self.actor_noised.load_state_dict(self.actor.state_dict().copy())
                # add noise to the copy
                self.actor_noised.add_parameter_noise(self.scalar)
                # get the next action values from the noised actor
                action_noised = self.actor_noised.forward(state).cpu().data.numpy()
                # meassure the distance between the action values from the regular and 
                # the noised actor to adjust the amount of noise that will be added next round
                distance = np.sqrt(np.mean(np.square(actions-action_noised)))
                # for stats and print only
                self.distances.append(distance)
                # adjust the amount of noise given to the actor_noised
                if distance > self.desired_distance:
                    self.scalar *= self.scalar_decay
                if distance < self.desired_distance:
                    self.scalar /= self.scalar_decay
                # set the noised action as action
                action = action_noised

            elif self.noise_type == "ou":
                action = actions + self.ou_noise()
            else:
                action = actions + np.random.randn(self.n_actions) * self.normal_scalar


        return np.clip(action, -np.pi, np.pi)
    
    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau

        target_actor_params = self.target_actor.named_parameters()
        actor_params = self.actor.named_parameters()

        target_actor_state_dict = dict(target_actor_params)
        actor_state_dict = dict(actor_params)
        for name in actor_state_dict:
            actor_state_dict[name] = tau*actor_state_dict[name].clone() + \
                    (1-tau)*target_actor_state_dict[name].clone()

        self.target_actor.load_state_dict(actor_state_dict)

        target_critic_params = self.target_critic.named_parameters()
        critic_params = self.critic.named_parameters()

        target_critic_state_dict = dict(target_critic_params)
        critic_state_dict = dict(critic_params)
        for name in critic_state_dict:
            critic_state_dict[name] = tau*critic_state_dict[name].clone() + \
                    (1-tau)*target_critic_state_dict[name].clone()

        self.target_critic.load_state_dict(critic_state_dict)

    def save_models(self):
        self.actor.save_checkpoint()
        self.target_actor.save_checkpoint()
        self.critic.save_checkpoint()
        self.target_critic.save_checkpoint()

    def load_models(self):
        self.actor.load_checkpoint()
        self.target_actor.load_checkpoint()
        self.critic.load_checkpoint()
        self.target_critic.load_checkpoint()

    def __str__(self):
        return f"\n{'#'*80}\n\nHyperparameters: \n\n{self.hyperparameters}\n\n{self.actor}\n\n{self.critic}\n\n{'#'*80}\n\n"
