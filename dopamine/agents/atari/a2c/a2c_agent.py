
import os
import sys
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical


class Net(nn.Module):

    def __init__(self, num_actions):
        super().__init__()
        orthogonal = nn.init.orthogonal_
        self.conv1 = nn.Conv2d(4, 32, 8, stride=4)
        orthogonal(self.conv1.weight)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        orthogonal(self.conv2.weight)
        self.conv3 = nn.Conv2d(64, 64, 3, stride=1)
        orthogonal(self.conv3.weight)
        self.pool = nn.AdaptiveAvgPool2d(1)
        # self.fc = nn.Linear(3136, 512)
        self.fc = nn.Linear(64, 256)
        orthogonal(self.fc.weight)
        self.logits = nn.Linear(256, num_actions)
        orthogonal(self.logits.weight)
        self.value = nn.Linear(256, 1)
        orthogonal(self.value.weight)

    def forward(self, x):
        x = x / 255.
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc(x))
        logits = self.logits(x)
        value = self.value(x)
        return logits, value


class A2CAgent(object):

    def __init__(self,
                 num_actions,
                 n_env, 
                 gamma=0.99,
                 train_period=5, 
                 v_loss_coef=0.5,
                 entropy_coef=0.01,
                 torch_device='cuda'):
        self.num_actions = num_actions
        self.n_env = n_env
        self.gamma = gamma
        self.train_period = train_period
        self.v_loss_coef = v_loss_coef
        self.entropy_coef = entropy_coef
        self.torch_device = torch_device

        self.net = Net(num_actions).to(self.torch_device)
        self.optimizer = torch.optim.RMSprop(self.net.parameters(), 
                                             lr=0.0007,
                                             alpha=0.99)

        self.eval_mode = False

        self.step_num = 0
        
    def _select_action(self, obs):
        if self.eval_mode:
            obs = obs[None, :]
        obs = torch.tensor(obs, dtype=torch.float32, device=self.torch_device)
        logits, value = self.net(obs)
        m = Categorical(logits=logits)
        action = m.sample()
        action = action.tolist()
        return action
    
    def begin_episode(self, obs):
        action = self._select_action(obs)
        return action

    def step(self, obs):
        action = self._select_action(obs)
        return action

    def train(self, env, min_steps):
        train_steps  = 0
        obs = env.reset()
        while train_steps < min_steps:
            obs, batch_obs, batch_action, batch_discount_reward = self._collect_information(env, obs)
            batch_discount_reward = torch.tensor(batch_discount_reward, dtype=torch.float32, device=self.torch_device).view(-1)
            batch_action = torch.tensor(batch_action, dtype=torch.int32, device=self.torch_device).view(-1)
            batch_obs = torch.tensor(batch_obs, dtype=torch.float32, device=self.torch_device)
            
            batch_logits, batch_v = self.net(batch_obs)
            batch_v = batch_v.view(-1)

            batch_advantage = batch_discount_reward - batch_v

            m = Categorical(logits=batch_logits)
            entropy = torch.mean(m.entropy())
            log_probs = m.log_prob(batch_action)
            pg_loss = - torch.mean(log_probs * batch_advantage.detach())
            v_loss = torch.mean(torch.pow((batch_discount_reward.detach() - batch_v), 2))

            loss = pg_loss + self.v_loss_coef * v_loss - self.entropy_coef * entropy

            if train_steps % 100 == 0:
                sys.stdout.write(f'entropy: {entropy} pg_loss: {pg_loss} v_loss: {v_loss} total_loss: {loss}\r')
                sys.stdout.flush()

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            train_steps += self.n_env * self.train_period

    def _collect_information(self, env, obs):
        obs_buffer, action_buffer, reward_buffer, terminal_buffer = [], [], [], []
        for _ in range(self.train_period):
            obs_buffer.append(obs)
            action = self._select_action(obs)
            obs, reward, terminal, info = env.step(action)
            action_buffer.append(action)
            reward_buffer.append(reward)
            terminal_buffer.append(terminal)

        batch_obs = np.asarray(obs_buffer, dtype=np.float32).swapaxes(0, 1).reshape(-1, 4, 84, 84)
        batch_reward = np.asarray(reward_buffer, dtype=np.float32).swapaxes(0, 1)
        batch_action = np.asarray(action_buffer, dtype=np.int32).swapaxes(0, 1)
        batch_terminal = np.asarray(terminal_buffer, dtype=np.bool).swapaxes(0, 1)

        most_recent_obs = torch.tensor(obs, dtype=torch.float32, device=self.torch_device)
        _, most_recent_value = self.net(most_recent_obs)
        most_recent_value = most_recent_value.cpu().detach().numpy()

        batch_discount_reward = self.compute_discount_rewards(batch_reward, batch_terminal, most_recent_value)
      
        return obs, batch_obs, batch_action, batch_discount_reward
    
    def _train_step(self, obs):
        batch_obs = np.asarray(self.obs_buffer, dtype=np.float32).swapaxes(0, 1).reshape(-1, 4, 84, 84)
        batch_reward = np.asarray(self.reward_buffer, dtype=np.float32).swapaxes(0, 1)
        batch_action = np.asarray(self.action_buffer, dtype=np.int32).swapaxes(0, 1)
        batch_terminal = np.asarray(self.terminal_buffer, dtype=bool).swapaxes(0, 1)

        obs = torch.tensor(obs, dtype=torch.float32, device=self.torch_device)
        _, last_value = self.net(obs)
        last_value = last_value.view(-1).cpu().detach().numpy()

        batch_discount_reward = self.compute_discount_rewards(batch_reward, batch_terminal, last_value)
       
        batch_discount_reward = torch.tensor(batch_discount_reward, dtype=torch.float32, device=self.torch_device).view(-1)
        batch_action = torch.tensor(batch_action, dtype=torch.float32, device=self.torch_device).view(-1)
        batch_obs = torch.from_numpy(batch_obs).to(self.torch_device)
        
        batch_logits, batch_v = self.net(batch_obs)
        batch_v = batch_v.view(-1)

        batch_advantage = batch_discount_reward - batch_v

        m = Categorical(logits=batch_logits)
        entropy = torch.mean(m.entropy())
        log_probs = m.log_prob(batch_action)
        pg_loss = - torch.mean(log_probs * batch_advantage.detach())
        v_loss = torch.mean(torch.pow((batch_discount_reward.detach() - batch_v), 2))

        loss = pg_loss + self.v_loss_coef * v_loss - self.entropy_coef * entropy

        if self.step_num % 100 == 0:
            sys.stdout.write(f'entropy: {entropy} pg_loss: {pg_loss} v_loss: {v_loss} total_loss: {loss}\r')
            sys.stdout.flush()
 
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=0.5)
            
        self.optimizer.step()

    def _swap_and_flatten(self, array):
        shape = array.shape
        return array.swapaxes(0, 1).reshape(shape[0] * shape[1], *shape[2:])

    def compute_discount_rewards(self, batch_reward, batch_terminal, last_value):
        batch_discount_reward = np.zeros_like(batch_reward)
        for n, (reward, terminal, value) in enumerate(zip(batch_reward, batch_terminal, last_value)):
            discount_reward = []
            R = value
            for r, t in zip(reward[::-1], terminal[::-1]):
                R = r + self.gamma * R * (1 - t)
                discount_reward.insert(0, R)
            batch_discount_reward[n] = discount_reward
        return batch_discount_reward

    def bundle_and_checkpoint(self, checkpoint_dir, iteration_number):
        if not os.path.exists(checkpoint_dir):
            return None
        
        torch.save(self.net.state_dict(), os.path.join(checkpoint_dir, 'net_ckpt-{}'.format(iteration_number)))
        torch.save(self.optimizer.state_dict(), os.path.join(checkpoint_dir, 'opt_ckpt-{}'.format(iteration_number)))

        bundle_dict = {}
        return bundle_dict
    
    def unbundle(self, checkpoint_dir, iteration_number, bundle_dict):

        for key in self.__dict__:
            if key in bundle_dict:
                self.__dict__[key] = bundle_dict[key]

        self.net.load_state_dict(torch.load(os.path.join(checkpoint_dir, 'net_ckpt-{}'.format(iteration_number))))
        self.optimizer.load_state_dict(torch.load(os.path.join(checkpoint_dir, 'opt_ckpt-{}'.format(iteration_number))))
        return True


        
        