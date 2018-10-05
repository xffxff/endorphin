
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical


class Net(nn.Module):

    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 128)
        self.logits = nn.Linear(128, 2)
        self.value = nn.Linear(128, 1)

    def forward(self, x):
        x = torch.relu(self.fc(x))
        logits = self.logits(x)
        value = self.value(x)
        return logits, value


class A2CAgent(object):

    def __init__(self,
                 num_actions, 
                 gamma=0.99,
                 torch_device='cpu'):
        self.num_actions = num_actions
        self.gamma = gamma
        self.torch_device = torch_device

        self.net = Net()
        self.optimizer = torch.optim.Adam(self.net.parameters())

        self.eval_mode = False

        self.step_num = 0
        # self.log_prob_for_action = []
        self.obs_buffer = []
        self.action_buffer = []
        self.reward_buffer = []
        self.terminal_buffer = []
        
    def _select_action(self, obs):
        obs = torch.tensor(obs, dtype=torch.float32)
        logits, value = self.net(obs)
        m = Categorical(logits=logits)
        action = m.sample()
        if self.eval_mode:
            action = action.item()
        else:
            action = action.tolist()
        return action
    
    def begin_episode(self, obs):
        self.obs_buffer.append(obs)
        action = self._select_action(obs)
        self.action_buffer.append(action)
        return action

    def step(self, reward, obs, terminal):
        self.reward_buffer.append(reward)
        self.terminal_buffer.append(terminal)

        self.step_num += 1

        if self.step_num % 5 == 0:
            if not self.eval_mode:
                self._train_step(obs)
            del self.obs_buffer[:]
            del self.action_buffer[:]
            del self.reward_buffer[:]
            del self.terminal_buffer[:]
        
        self.obs_buffer.append(obs)
        action = self._select_action(obs)
        self.action_buffer.append(action)
        return action
    
    def _train_step(self, obs):
        batch_obs = np.asarray(self.obs_buffer).swapaxes(0, 1).reshape(-1, 4)
        batch_reward = np.asarray(self.reward_buffer, dtype=np.float32).swapaxes(0, 1)
        batch_action = np.asarray(self.action_buffer, dtype=np.int32).swapaxes(0, 1)
        # batch_value = np.asarray(self.value_buffer, dtype=np.float32).swapaxes(0, 1)
        batch_terminal = np.asarray(self.terminal_buffer, dtype=bool).swapaxes(0, 1)

        obs = torch.tensor(obs, dtype=torch.float32)
        _, last_value = self.net(obs)
        last_value = last_value.view(-1).detach().numpy()

        batch_target_v = []
        for n, (reward, terminal, value) in enumerate(zip(batch_reward, batch_terminal, last_value)):
            target_v = []
            R = value
            for r, t in zip(reward[::-1], terminal[::-1]):
                R = r + self.gamma * R * (1 - t)
                target_v.insert(0, R)
            batch_target_v.append(target_v)
        batch_target_v = torch.tensor(batch_target_v, dtype=torch.float32).view(-1)
        batch_action = torch.tensor(batch_action, dtype=torch.float32).view(-1)
        batch_obs = torch.tensor(batch_obs, dtype=torch.float32)
        
        batch_logits, batch_v = self.net(batch_obs)
        batch_v = batch_v.view(-1)

        batch_advantage = batch_target_v - batch_v

        m = Categorical(logits=batch_logits)
        self.entopy = torch.mean(m.entropy())
        log_probs = m.log_prob(batch_action)
        self.pg_loss = - torch.mean(log_probs * batch_advantage.detach())
        self.v_loss = torch.mean(torch.pow((batch_target_v.detach() - batch_v), 2))

        loss = self.pg_loss + 0.25 * self.v_loss - 0.01 * self.entopy

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


        
        