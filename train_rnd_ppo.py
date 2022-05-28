import minerl
import gym
import argparse
from pyrsistent import T
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
from utils import make_11action, save_model, converter
from models.PPO import PPO, Buffer, train_net

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

MODEL_NAME = "RND_PPO"
ENV_NAME = 'MineRLTreechop-v0'
SAVE_PATH = "C:/Users/ye200/OneDrive/Desktop/MineRL_RND/weights/" + MODEL_NAME + '/'
SUMMARY_PATH = "C:/Users/ye200/OneDrive/Desktop/MineRL_RND/experiments/"+ MODEL_NAME + '/'
paths = [SUMMARY_PATH, SAVE_PATH]
for path in paths:
    if not os.path.isdir(path):
        os.mkdir(path)
        

#Hyperparameters
BATCH_SIZE = 128   # 32S
LR = 0.0005
GAMMA         = 0.999
LMBDA         = 0.95
EPS_CLIP      = 0.1
K_EPOCH       = 10
T_HORIZON     = 20
RND_START = int(0)
INT_GAMMA = 0.99
SAVE_PERIOD = 1000
ENT_COEF = 1e-2
MAX_GRAD_NORM = 0.1
total_episodes = 100000

def compute_int_reward(rnd_target_net, rnd_pred_net, next_state):
    target = rnd_target_net(next_state)
    prediction = rnd_pred_net(next_state)
    return F.mse_loss(target, prediction)

def main():
    writer = SummaryWriter(SUMMARY_PATH)
    env = gym.make(ENV_NAME)
    env.make_interactive(port=6666, realtime=False)
    model = PPO(num_actions=11).to(device)
    # ! RND Init
    rnd_target_net = RNDNetwork()
    rnd_predictor_net = RNDNetwork()

    buffer = Buffer(T_HORIZON)
    optimizer = optim.Adam(model.parameters(), lr=LR)

    for n_epi in range(total_episodes):
        score = 0.0
        s = env.reset()
        done = False
        loss = 0.0
        while not done:
            for t in range(T_HORIZON):
                prob = model.pi(converter(s, device), softmax_dim=1).squeeze(0)
                m = Categorical(prob)
                action = m.sample().item()
                action_dict = make_11action(env, action)
                s_prime, reward, done, info = env.step(action_dict)

                # TODO : 
                int_reward = compute_int_reward(converter(s_prime, device))

                buffer.put_data((converter(s, device), action, reward, converter(s_prime, device), prob[action].item(), done))
                s = s_prime
                score += reward

                if done:
                    break
            if done:
                writer.add_scalar("total_rewards", score, n_epi)
                writer.add_scalar("loss", loss, n_epi)
                print("# of episode :{}, score : {:.1f}".format(n_epi, score))
                break
                   
            loss = train_net(model, buffer, optimizer, K_EPOCH, LMBDA, GAMMA, EPS_CLIP)
        if n_epi % 100 == 0:
            save_model(n_epi, SAVE_PERIOD, SAVE_PATH, model, MODEL_NAME, ENV_NAME)

            
    env.close()

if __name__ == '__main__':
    main()