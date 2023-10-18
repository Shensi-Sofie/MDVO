import numpy as np
import torch
import collections
import random


class ReplayBuffer():
    def __init__(self, buffer_limit, state_dim, action_space, device):
        self.buffer = collections.deque(maxlen=buffer_limit)
        self.state_dim = state_dim
        self.action_space = action_space
        self.device = device

    def put(self, transition):
        self.buffer.append(transition)

    def sample(self, n):
        mini_batch = random.sample(self.buffer, n)
        state_lst, reward_lst, next_state_lst, done_mask_lst = [], [], [], []
        actions_lst = [[] for i in range(self.action_space)]

        for transition in mini_batch:
            state, actions, reward, next_state, done_mask = transition
            state = list(np.reshape(state, (self.state_dim,)))
            next_state = list(np.reshape(next_state, (self.state_dim,)))
            state_lst.append(state)
            for idx in range(self.action_space):
                actions_lst[idx].append(actions[idx])
            reward_lst.append(reward)
            next_state_lst.append(next_state)
            done_mask_lst.append([done_mask])
        reward_lst = np.array(reward_lst, dtype='float32')
        reward_lst = list(np.reshape(reward_lst, (n, 1)))
        actions_lst = [torch.tensor(x, dtype=torch.float).to(self.device) for x in actions_lst]
        return torch.tensor(state_lst, dtype=torch.float).to(self.device), \
               actions_lst, \
               torch.tensor(reward_lst).to(self.device), \
               torch.tensor(next_state_lst, dtype=torch.float).to(self.device), \
               torch.tensor(done_mask_lst).to(self.device)

    def size(self):
        return len(self.buffer)


W_DD = 1
W_DE = 4
W_RE = 8
d_trans_power = 4
e_trans_power = 10
WHITE_NOISE = 10e-14
e_pc = 10e7
d_pc = 10e4
STEPDUR = 0.1
DATA_SIZE = 1
DATA_PRO_DENSITY = 100

# D-D
def dd_trans_t():
    x = d_trans_power / WHITE_NOISE * (100 ** 4)
    rate = W_DD * np.log2(1 + x)
    trans_t = DATA_SIZE / rate
    return trans_t


def d_pro_t():
    return DATA_SIZE * DATA_PRO_DENSITY/d_pc


def d_pro_num():
    return int((STEPDUR - dd_trans_t()) / d_pro_t())

#D-E
def de_trans_t():
    x = d_trans_power/WHITE_NOISE*(100**4)
    rate = W_DE * np.log2(1+x)
    trans_t = DATA_SIZE/rate
    return trans_t


def e_pro_t():
    return DATA_SIZE * DATA_PRO_DENSITY/e_pc


def e_pro_num():
    return int((STEPDUR-de_trans_t()) / e_pro_t())



def moving_average(raw_list):
    new_list = []
    for i in range(len(raw_list)):
        if i == 0:
            new_list.append(raw_list[i])
        else:
            new_list.append(new_list[i-1]*0.9 + raw_list[i]*0.1)
    return new_list


def test_delay(new_task, data_size):
    compute_delay = new_task/ (e_pc / 2)   # ES的运算能力10e5MHz
    x = d_trans_power / WHITE_NOISE * (100 ** 4)
    rate = W_DE * np.log2(1 + x)
    trans_delay = data_size / rate
    delay = compute_delay + trans_delay
    return delay

