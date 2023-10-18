import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import numpy as np
from network import QNetwork


class DDQN(nn.Module):
    def __init__(self, state_space: int, action_num: int, action_scale: int, learning_rate, device: str):
        super(DDQN,self).__init__()
        self.action_scale = action_scale
        self.action_num = action_num
        self.q = QNetwork(state_space, action_num, action_scale).to(device)
        self.target_q = QNetwork(state_space, action_num, action_scale).to(device)
        self.target_q.load_state_dict(self.q.state_dict())

        self.optimizer = optim.Adam([{'params': self.q.linear_1.parameters(),'lr': learning_rate / (action_num+2)},\
                                    {'params': self.q.linear_2.parameters(),'lr': learning_rate / (action_num+2)},\
                                    {'params': self.q.value.parameters(), 'lr': learning_rate/ (action_num+2)},\
                                    {'params': self.q.actions.parameters(), 'lr': learning_rate}])
        self.update_freq = 1000
        self.update_count = 0

    def action(self, obs, epsilon):
        if epsilon > random.random():
            action = np.random.choice(range(0, self.action_scale), self.action_num)
        else:
            action_prob = self.q(torch.tensor(obs).float().reshape(1, -1).to('cpu'))
            action = [int(x.max(1)[1]) for x in action_prob]
        return action
    
    def train_mode(self, memory, batch_size, gamma):
        state, actions, reward, next_state, done_mask = memory.sample(batch_size)
        actions = torch.stack(actions).transpose(0, 1).unsqueeze(-1)
        done_mask = torch.abs(done_mask - 1)
        
        cur_actions = self.q(state)
        cur_actions = torch.stack(cur_actions).transpose(0, 1)
        cur_actions = cur_actions.gather(2, actions.long()).squeeze(-1)

        target_cur_actions = self.target_q(next_state)
        target_cur_actions = torch.stack(target_cur_actions).transpose(0, 1)
        target_cur_actions = target_cur_actions.max(-1, keepdim=True)[0]
        target_action = (done_mask * gamma * target_cur_actions.mean(1) + reward)
        
        loss = F.mse_loss(cur_actions, target_action.repeat(1, self.action_num))

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        self.update_count += 1
        if (self.update_count % self.update_freq == 0) and (self.update_count > 0):
            self.update_count = 0
            self.target_q.load_state_dict(self.q.state_dict())
        return loss
