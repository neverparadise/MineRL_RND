import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from collections import deque

# Hyperparameters
LR = 0.000625

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
class PPO(nn.Module):
    def __init__(self, num_actions):
        super(PPO, self).__init__()
        self.num_actions = num_actions
        
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
    
    def forward(self, x, softmax_dim=1):
        if len(x.shape) < 4:
          x = x.unsqueeze(0)
        conv_feature = self.conv_layers(x)
        prob = self.fc_pi(conv_feature)
        log_prob = F.log_softmax(prob, dim=softmax_dim)
        value = self.fc_v(conv_feature)
        return log_prob, value
class Buffer:
    def __init__(self, T_horizon):
        self.data = deque(maxlen=T_horizon)

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
        
def train_net(model, buffer, optimizer, K_epoch, lmbda, gamma, eps_clip):
    s, a, r, s_prime, done_mask, prob_a = buffer.make_batch()

    for i in range(K_epoch):
        td_target = r + gamma * model.v(s_prime) * done_mask
        delta = td_target - model.v(s)
        delta = delta.detach().cpu().numpy()

        advantage_lst = []
        advantage = 0.0
        for delta_t in delta[::-1]:
            advantage = gamma * lmbda * advantage + delta_t[0]
            advantage_lst.append([advantage])
        advantage_lst.reverse()
        advantage = torch.tensor(advantage_lst, dtype=torch.float).cuda()

        pi = model.pi(s, softmax_dim=1)
        pi_a = pi.gather(1,a)
        ratio = torch.exp(torch.log(pi_a) - torch.log(prob_a))  # a/b == exp(log(a)-log(b))

        surr1 = ratio * advantage
        surr2 = torch.clamp(ratio, 1-eps_clip, 1+eps_clip) * advantage
        loss = -torch.min(surr1, surr2) + F.smooth_l1_loss(model.v(s) , td_target.detach())

        optimizer.zero_grad()
        loss.mean().backward()
        optimizer.step()
    
    return loss.mean().item()
