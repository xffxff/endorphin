
import os
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical


class Net(nn.Module):

    def __init__(self, num_actions):
        super().__init__()
        self.fc = nn.Linear(4, 128)
        self.logits = nn.Linear(128, num_actions)
        self.value = nn.Linear(128, 1)

    def forward(self, x):
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
                 v_loss_coef=0.25,
                 entropy_coef=0.01,
                 torch_device='cpu'):
        self.num_actions = num_actions
        self.n_env = n_env
        self.gamma = gamma
        self.train_period = train_period
        self.v_loss_coef = v_loss_coef
        self.entropy_coef = entropy_coef
        self.torch_device = torch_device

        self.net = Net(num_actions).to(self.torch_device)
        self.optimizer = torch.optim.Adam(self.net.parameters())

        self.eval_mode = False

        self.step_num = 0
        
    def select_action(self, obs):
        """Select an action from the set of available actions.

        Returns:
            int, the selected action.
        """
        obs = torch.tensor(obs, dtype=torch.float32, device=self.torch_device)
        logits, value = self.net(obs)
        m = Categorical(logits=logits)
        action = m.sample()
        if self.eval_mode:
            action = action.item()
        else:
            action = action.tolist()
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
            obs, batch_obs, batch_action, batch_discount_reward = self._collect_information(env, obs)
            batch_discount_reward = torch.tensor(batch_discount_reward, dtype=torch.float32, device=self.torch_device)
            batch_action = torch.tensor(batch_action, dtype=torch.int32, device=self.torch_device)
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

            self.optimizer.zero_grad()
            loss.backward()

            # I find clip grap norm make performance worse
            # nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=1)
            self.optimizer.step()

            train_steps += self.n_env * self.train_period

    def _collect_information(self, env, obs):
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
        for _ in range(self.train_period):
            obs_buffer.append(obs)
            action = self.select_action(obs)
            obs, reward, terminal, info = env.step(action)
            action_buffer.append(action)
            reward_buffer.append(reward)
            terminal_buffer.append(terminal)
        obs_buffer = np.asarray(obs_buffer, dtype=obs.dtype)
        action_buffer = np.asarray(action_buffer)
        reward_buffer = np.asarray(reward_buffer, dtype=np.float32)
        terminal_buffer = np.asarray(terminal_buffer, dtype=np.bool)

        most_recent_obs = torch.tensor(obs, dtype=torch.float32, device=self.torch_device)
        _, most_recent_value = self.net(most_recent_obs)
        most_recent_value = most_recent_value.cpu().detach().numpy()
        discount_reward_buffer = np.zeros_like(reward_buffer)
        next_reward = most_recent_value.squeeze()
        for step in reversed(range(self.train_period)):
            next_reward = reward_buffer[step] + self.gamma * (1 - terminal_buffer[step]) * next_reward
            discount_reward_buffer[step] = next_reward

        obs_buffer, action_buffer, discount_reward_buffer = \
            map(self._swap_and_flatten, (obs_buffer, action_buffer, discount_reward_buffer))
        return obs, obs_buffer, action_buffer, discount_reward_buffer

    def _swap_and_flatten(self, array):
        shape = array.shape
        return array.swapaxes(0, 1).reshape(shape[0] * shape[1], *shape[2:])

    def bundle_and_checkpoint(self, checkpoint_dir, iteration_number):
        """Returns a self-contained bundle of the agent's state.

        Args:
            checkpoint_dir: str, directory where TensorFlow objects will be saved.
            iteration_number: int, iteration number to use for naming the checkpoint file.

        Returns:
            A dict containing additional Python objects to be checkpointed by the
            experiment. If the checkpoint directory does not exist, returns None.
        """
        if not os.path.exists(checkpoint_dir):
            return None
        
        torch.save(self.net.state_dict(), os.path.join(checkpoint_dir, 'net_ckpt-{}'.format(iteration_number)))
        torch.save(self.optimizer.state_dict(), os.path.join(checkpoint_dir, 'opt_ckpt-{}'.format(iteration_number)))

        bundle_dict = {}
        return bundle_dict
    
    def unbundle(self, checkpoint_dir, iteration_number, bundle_dict):
        """Restores the agent from a checkpoint.

        Restores the agent's Python objects to those specified in bundle_dictionary,
        and restores the TensorFlow objects to those specified in the
        checkpoint_dir. If the checkpoint_dir does not exist, will not reset the
        agent's state.

        Args:
        checkpoint_dir: str, path to the checkpoint saved by tf.Save.
        iteration_number: int, checkpoint version, used when restoring replay
            buffer.
        bundle_dictionary: dict, containing additional Python objects owned by
            the agent.

        Returns:
        bool, True if unbundling was successful.
        """
        for key in self.__dict__:
            if key in bundle_dict:
                self.__dict__[key] = bundle_dict[key]

        self.net.load_state_dict(torch.load(os.path.join(checkpoint_dir, 'net_ckpt-{}'.format(iteration_number))))
        self.optimizer.load_state_dict(torch.load(os.path.join(checkpoint_dir, 'opt_ckpt-{}'.format(iteration_number))))
        return True


        
        