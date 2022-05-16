import minerl
import gym
import argparse
import torch
import torch.optim as optim
import torch.nn as nn
from torch.distributions import Categorical
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
import numpy as np
import yaml
import os
from copy import deepcopy
import argparse

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def make_11action(env, action_index, always_attack=True):
    # Action들을 정의
    action = env.action_space.noop()
    # Cameras
    if (action_index == 0):
        action['camera'] = [0, 0]
    elif (action_index == 1):
        action['camera'] = [0, -5]
    elif (action_index == 2):
        action['camera'] = [0, 5]
    elif (action_index == 3):
        action['camera'] = [-5, 0]
    elif (action_index == 4):
        action['camera'] = [5, 0]

    # Forwards
    elif (action_index == 5):
        action['forward'] = 0
    elif (action_index == 6):
        action['forward'] = 1

    # Jump
    elif (action_index == 7):
        action['jump'] = 0
    elif (action_index == 8):
        action['jump'] = 1

    # Attack 
    elif (action_index == 9):
        action['attack'] = 0
    elif (action_index == 10):
        action['attack'] = 1
    
    return action

def save_model(model):
    torch.save({'model_state_dict': model.state_dict()}, './PPO.pth')
    print("model saved")


def converter(observation, device):
    obs = observation['pov']
    obs = obs / 255.0
    obs = torch.from_numpy(obs)
    obs = obs.permute(2, 0, 1)
    return obs.float().to(device)

#Hyperparameters
learning_rate = 0.0005
gamma         = 0.99
lmbda         = 0.95
eps_clip      = 0.1
K_epoch       = 3
T_horizon     = 20

class PPO(nn.Module):
    def __init__(self, num_actions):
        super(PPO, self).__init__()
        self.num_actions = num_actions
        self.data = []
        
        self.conv_layers = nn.Sequential(
        nn.Conv2d(3, 32, kernel_size=8, stride=4),
        nn.BatchNorm2d(32),
        nn.LeakyReLU(),
        nn.Conv2d(32, 64, kernel_size=4, stride=2),
        nn.BatchNorm2d(64),
        nn.LeakyReLU(),
        nn.Conv2d(64, 64, kernel_size=3, stride=1),
        nn.BatchNorm2d(64),
        nn.LeakyReLU(),
        nn.Flatten()
        )

        def conv2d_size_out(size, kernel_size=3, stride=2):
            return (size - (kernel_size - 1) - 1) // stride + 1

        conv_size = conv2d_size_out(64, 8, 4)
        conv_size = conv2d_size_out(conv_size, 4, 2)
        conv_size = conv2d_size_out(conv_size, 3, 1)
        linear_input_size = conv_size * conv_size * 64 # 4 x 4 x 64 = 1024

        self.fc_pi = nn.Linear(linear_input_size, self.num_actions)
        self.fc_v = nn.Linear(linear_input_size, 1)
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    def pi(self, x, softmax_dim=0):
        if len(x.shape) < 4:
          x = x.unsqueeze(0)
        x = self.conv_layers(x)
        x = self.fc_pi(x)
        prob = F.softmax(x, dim=softmax_dim)
        return prob
    
    def v(self, x):
        if len(x.shape) < 4:
          x = x.unsqueeze(0)
        x = self.conv_layers(x)
        v = self.fc_v(x)
        return v
      
    def put_data(self, transition):
        self.data.append(transition)
        
    def make_batch(self):
        s_lst, a_lst, r_lst, s_prime_lst, prob_a_lst, done_lst = [], [], [], [], [], []
        for transition in self.data:
            s, a, r, s_prime, prob_a, done = transition
            
            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            prob_a_lst.append([prob_a])
            done_mask = 0 if done else 1
            done_lst.append([done_mask])
            
        s = torch.stack(s_lst).to(device)
        a = torch.tensor(a_lst, dtype=torch.int64).to(device)
        r = torch.tensor(r_lst, dtype=torch.float).to(device)
        s_prime =  torch.stack(s_prime_lst).to(device)
        done_mask = torch.tensor(done_lst, dtype=torch.float).to(device)
        prob_a = torch.tensor(prob_a_lst).to(device)
        self.data = []
        return s, a, r, s_prime, done_mask, prob_a
        
    def train_net(self):
        s, a, r, s_prime, done_mask, prob_a = self.make_batch()

        for i in range(K_epoch):
            td_target = r + gamma * self.v(s_prime) * done_mask
            delta = td_target - self.v(s)
            delta = delta.detach().cpu().numpy()

            advantage_lst = []
            advantage = 0.0
            for delta_t in delta[::-1]:
                advantage = gamma * lmbda * advantage + delta_t[0]
                advantage_lst.append([advantage])
            advantage_lst.reverse()
            advantage = torch.tensor(advantage_lst, dtype=torch.float).cuda()

            pi = self.pi(s, softmax_dim=1)
            pi_a = pi.gather(1,a)
            ratio = torch.exp(torch.log(pi_a) - torch.log(prob_a))  # a/b == exp(log(a)-log(b))

            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1-eps_clip, 1+eps_clip) * advantage
            loss = -torch.min(surr1, surr2) + F.smooth_l1_loss(self.v(s) , td_target.detach())

            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()



total_episodes = 5
print_interval = 20

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    env = gym.make("MineRLTreechop-v0")
    env.make_interactive(port=6666, realtime=False)
    model = PPO(num_actions=11).to(device)


    for n_epi in range(total_episodes):
        score = 0.0
        s = env.reset()
        done = False
        while not done:
            for t in range(T_horizon):
                prob = model.pi(converter(s, device), softmax_dim=1).squeeze(0)
                m = Categorical(prob)
                a = m.sample().item()
                action = make_11action(env, a)
                s_prime, r, done, info = env.step(action)

                model.put_data((converter(s, device), a, r, converter(s_prime, device), prob[a].item(), done))
                s = s_prime
                score += r
                if done:
                    print("# of episode :{}, score : {:.1f}".format(n_epi, score))
                    break

            model.train_net()
    env.close()
    save_model(model)

if __name__ == '__main__':
    main()