import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from collections import deque

class RND(nn.Module):
    def __init__(self,):
        super(RND, self).__init__()