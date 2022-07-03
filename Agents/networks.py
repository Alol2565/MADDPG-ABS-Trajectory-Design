import os
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)

class CriticNetwork(nn.Module):
    def __init__(self, beta, input_dims, fc1_dims, fc2_dims, fc3_dims, fc4_dims, fc5_dims,
                    n_agents, n_actions, name, chkpt_dir):
        super(CriticNetwork, self).__init__()

        self.chkpt_file = os.path.join(chkpt_dir, name)
    
        self.fc1 = nn.Linear(input_dims+n_agents*n_actions, fc1_dims)
        # self.bn1d = nn.BatchNorm1d(fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.fc3 = nn.Linear(fc2_dims, fc3_dims)
        self.fc4 = nn.Linear(fc3_dims, fc4_dims)
        # self.fc5 = nn.Linear(fc4_dims, fc5_dims)
        self.q = nn.Linear(fc4_dims, 1)

        self.optimizer = optim.Adam(self.parameters(), lr=beta)
        # self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.999)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=100, gamma=0.01)
        # self.scheduler_chpt = optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.95)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
 
        self.to(self.device)

        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(*hidden_init(self.fc3))
        self.fc4.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state, action):
        x = F.relu((self.fc1(T.cat([state, action], dim=1))))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        # x = F.relu(self.fc5(x))
        q = self.q(x)

        return q

    def save_checkpoint(self):
        T.save(self.state_dict(), self.chkpt_file)
        # self.scheduler_chpt.step()

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.chkpt_file))


class ActorNetwork(nn.Module):
    def __init__(self, alpha, input_dims, fc1_dims, fc2_dims, fc3_dims, fc4_dims, fc5_dims,
                 n_actions, name, chkpt_dir):
        super(ActorNetwork, self).__init__()

        self.chkpt_file = os.path.join(chkpt_dir, name)

        self.fc1 = nn.Linear(input_dims, fc1_dims)
        # self.bn1d = nn.BatchNorm1d(fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.fc3 = nn.Linear(fc2_dims, fc3_dims)
        self.fc4 = nn.Linear(fc3_dims, fc4_dims)
        self.pi = nn.Linear(fc4_dims, n_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        # self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.999)
        # self.scheduler_chpt = optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.95)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=100, gamma=0.01)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
 
        self.to(self.device)

        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(*hidden_init(self.fc3))
        self.fc4.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state):
        x = F.relu((self.fc1(state)))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        pi = T.tanh(self.pi(x))
        return pi

    def add_parameter_noise(self, scalar=.1):
        self.fc1.weight.data += T.randn_like(self.fc1.weight.data) * scalar
        self.fc2.weight.data += T.randn_like(self.fc2.weight.data) * scalar
        self.fc3.weight.data += T.randn_like(self.fc3.weight.data) * scalar
        self.fc4.weight.data += T.randn_like(self.fc4.weight.data) * scalar

    def save_checkpoint(self):
        # self.scheduler.step()
        # print('lr: {0}'.format(self.optimizer.param_groups[0]['lr']))
        T.save(self.state_dict(), self.chkpt_file)
        # self.scheduler_chpt.step()

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.chkpt_file))