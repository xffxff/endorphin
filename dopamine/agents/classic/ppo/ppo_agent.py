
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical


class Net(nn.Module):

    def __init__(self, num_actions):
        super(Net, self).__init__()
        self.fc = nn.Linear(4, 128)
        self.logits = nn.Linear(128, num_actions)
        self.value = nn.Linear(128, 1)

    def forward(self, x):
        x = torch.relu(self.fc(x))
        logits = self.logits(x)
        value = self.value(x)
        return logits, value


class PPOAgent(object):
    """An implementation of PPO agent"""

    def __init__(self, 
                 num_actions,
                 gamma=0.99,
                 train_interval=128,
                 epoches=4, 
                 batch_steps=128,
                 v_loss_coef=0.25,
                 entropy_coef=0.01,
                 torch_device='cpu'):
        
        self.num_actions = num_actions
        self.gamma = gamma
        self.train_interval = train_interval
        self.epoches = epoches
        self.batch_steps = batch_steps
        self.v_loss_coef = v_loss_coef
        self.entropy_coef = entropy_coef
        self.torch_device = torch_device

        self.net = Net(num_actions)
        self.optimizer = torch.optim.Adam(self.net.parameters())

        self.eval_mode = False

        self.values = []
        self.log_probs = []
        
    def select_action(self, obs):
        """Select an action from the set of available actions.

        Returns:
            int, the selected action.
        """
        if self.eval_mode:
            obs = obs[None, :]
        obs = torch.tensor(obs, dtype=torch.float32, device=self.torch_device)
        logits, value = self.net(obs)
        m = Categorical(logits=logits)
        action = m.sample()

        if not self.eval_mode:
            self.values.append(value.cpu().detach().numpy())
            self.log_probs.append(m.log_prob(action).cpu().detach().numpy())
            action = action.tolist()
        else:
            action = action.item()
        
        return action

    def train(self, env, min_steps):
        """Train every regular steps.

        Args:
            env: the environment.
            min_steps: int, the minimum steps to execute.
        """
        train_steps  = 0
        obs = env.reset()
        while train_steps < min_steps:
            obs, obs_buffer, action_buffer, discount_reward_buffer = self._collect_experience(env, obs)

            ids = np.arange(self.train_interval)
            for epoch in range(self.epoches):
                np.random.shuffle(ids)
                for start in range(0, self.train_interval, self.batch_steps):
                    end = start + self.batch_steps
                    batch_ids = ids[start:end]
                    batch_obs, batch_action, batch_discount_reward, batch_old_value, batch_old_log_prob = \
                        (array[batch_ids] for array in (obs_buffer, action_buffer, discount_reward_buffer, self.values, self.log_probs))

                batch_discount_reward = torch.tensor(batch_discount_reward, dtype=torch.float32, device=self.torch_device)
                batch_action = torch.tensor(batch_action, dtype=torch.int32, device=self.torch_device)
                batch_obs = torch.tensor(batch_obs, dtype=torch.float32, device=self.torch_device)
                batch_old_value = torch.tensor(batch_old_value, dtype=torch.float32, device=self.torch_device)
                batch_old_log_prob = torch.tensor(batch_old_log_prob, dtype=torch.float32, device=self.torch_device)
            
                batch_logits, batch_value = self.net(batch_obs)
                batch_value = batch_value.view(-1)

                batch_advantage = (batch_discount_reward - batch_value).detach()

                m = Categorical(logits=batch_logits)
                entropy = torch.mean(m.entropy())
                batch_log_prob = m.log_prob(batch_action)
                ratio = torch.exp(batch_log_prob - batch_old_log_prob)
                pg_loss1 = - ratio * batch_advantage
                pg_loss2 = - batch_advantage * torch.clamp(ratio, 0.8, 1.2)
                pg_loss = torch.mean(torch.max(pg_loss1, pg_loss2))

                v_loss = torch.mean(torch.pow((batch_discount_reward.detach() - batch_value), 2))

                loss = pg_loss + self.v_loss_coef * v_loss - self.entropy_coef * entropy

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            train_steps += 4 * self.train_interval


    def _collect_experience(self, env, obs):
        """Collect experience when agent interact with the environment.

        Args:
            env: the environment the agent interact with.
            obs: numpy.ndarray, the most recent observation.
        Returns:
            obs: numpy.ndarray, the new most recent observation.
            batch_obs: numpy.ndarray, a series of observations.
            batch_action: numpy.ndarray, a series of actions.
            batch_discount_reward: numpy.ndarray, a series of rewards.
        """
        obs_buffer, action_buffer, reward_buffer, terminal_buffer = [], [], [], []
        self.log_probs, self.values = [], []
        for _ in range(self.train_interval):
            obs_buffer.append(obs)
            action = self.select_action(obs)
            obs, reward, terminal, info = env.step(action)
            action_buffer.append(action)
            reward_buffer.append(np.clip(reward, -1, 1))
            terminal_buffer.append(terminal)

        obs_buffer = np.asarray(obs_buffer, dtype=obs.dtype)
        reward_buffer = np.asarray(reward_buffer, dtype=np.float32)
        action_buffer = np.asarray(action_buffer)
        terminal_buffer = np.asarray(terminal_buffer, dtype=np.bool)
        self.log_probs = np.asarray(self.log_probs, dtype=np.float32)
        self.values = np.asarray(self.values, dtype=np.float32)

        most_recent_obs = torch.tensor(obs, dtype=torch.float32, device=self.torch_device)
        _, most_recent_value = self.net(most_recent_obs)
        most_recent_value = most_recent_value.cpu().detach().numpy()
        
        discount_reward_buffer = np.zeros_like(reward_buffer)
        next_reward = most_recent_value.squeeze()
        for step in reversed(range(self.train_interval)):
            next_reward = reward_buffer[step] + self.gamma * (1 - terminal_buffer[step]) * next_reward
            discount_reward_buffer[step] = next_reward

        obs_buffer, action_buffer, discount_reward_buffer, self.values, self.log_probs = \
            map(self._swap_and_flatten, (obs_buffer, action_buffer, discount_reward_buffer, self.values, self.log_probs))
        return obs, obs_buffer, action_buffer, discount_reward_buffer

    def _swap_and_flatten(self, array):
        shape = array.shape
        return array.swapaxes(0, 1).reshape(shape[0] * shape[1], *shape[2:])



    