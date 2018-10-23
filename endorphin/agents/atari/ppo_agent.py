
import os
import sys
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

from endorphin.net.conv import ConvNet


class PPOAgent(object):
    """An implementation of the A2C agnet"""

    def __init__(self,
                 num_actions,
                 n_env, 
                 gamma=0.99,
                 lam=0.95,
                 train_interval=128, 
                 epoches=4,
                 batch_steps=256,
                 clip_range=0.1,
                 v_loss_coef=0.5,
                 entropy_coef=0.01,
                 torch_device='cuda'):
        """Initialize the agent.

        Args:
            num_actions: int, number of the actions the agent can take at any state.
            gamma: float, discount factor with usual RL meaning.
            train_interval: int, the number of step the agent take every training.
            v_loss_coef: float, the coefficient of the value loss.
            entropy_coef: float, the coefficient of the entropy.
            torch_device: string, the Pytorch device on which the program is executed.
        """
        self.num_actions = num_actions
        self.n_env = n_env
        self.gamma = gamma
        self.lam = lam
        self.train_interval = train_interval
        self.epoches = epoches
        self.batch_steps = batch_steps
        self.clip_range = clip_range
        self.v_loss_coef = v_loss_coef
        self.entropy_coef = entropy_coef
        self.torch_device = torch_device

        self.net = ConvNet(num_actions).to(self.torch_device)
        self.optimizer = torch.optim.Adam(self.net.parameters(), 
                                             lr=0.00025,
                                             eps=1e-5)

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
            self.values.append(value.cpu().detach().numpy().squeeze())
            self.log_probs.append(m.log_prob(action).cpu().detach().numpy())

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
            obs, obs_buffer, action_buffer, discount_reward_buffer = self._collect_experience(env, obs)

            ids = np.arange(self.train_interval * self.n_env)
            for epoch in range(self.epoches):
                np.random.shuffle(ids)
                for start in range(0, self.train_interval * self.n_env, self.batch_steps):
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
                    m = Categorical(logits=batch_logits)
                    entropy = torch.mean(m.entropy())

                    batch_advantage = (batch_discount_reward - batch_old_value)
                    batch_advantage = (batch_advantage - batch_advantage.mean()) / (batch_advantage.std() + 1e-8)

                    batch_log_prob = m.log_prob(batch_action)

                    batch_clipped_value = batch_old_value + torch.clamp(batch_value - batch_old_value, - self.clip_range, self.clip_range)
                    v_loss1 = torch.pow(batch_discount_reward.detach() - batch_value, 2)
                    v_loss2 = torch.pow(batch_discount_reward.detach() - batch_clipped_value, 2)
                    v_loss = 0.5 * torch.mean(torch.max(v_loss1, v_loss2))
                    # v_loss = torch.mean(torch.pow(batch_discount_reward.detach() - batch_value, 2))

                    ratio = torch.exp(batch_log_prob - batch_old_log_prob)
                    pg_loss1 = - ratio * batch_advantage.detach()
                    pg_loss2 = - batch_advantage.detach() * torch.clamp(ratio, 1 - self.clip_range, 1 + self.clip_range)
                    pg_loss = torch.mean(torch.max(pg_loss1, pg_loss2))

                    loss = pg_loss + self.v_loss_coef * v_loss - self.entropy_coef * entropy

                    sys.stdout.write(f'steps: {train_steps}  entropy: {entropy:.3f}' \
                                     f'  pg_loss: {pg_loss:.6f}  v_loss: {v_loss:.6f}  total_loss: {loss:.6f}\r')
                    sys.stdout.flush()

                    self.optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.net.parameters(), 0.5)
                    self.optimizer.step()

            train_steps += self.n_env * self.train_interval

    def _collect_experience(self, env, obs):
        """Collect experience when agent interact with the environment.

        Args:
            env: the environment the agent interact with.
            obs: numpy.ndarray, the most recent observation.
        Returns:
            obs: numpy.ndarray, the new most recent observation.
            obs_buffer: numpy.ndarray, a series of observations.
            action_buffer: numpy.ndarray, a series of actions.
            discount_reward_buffer: numpy.ndarray, a series of rewards.
        """
        obs_buffer, action_buffer, reward_buffer, terminal_buffer = [], [], [], []
        self.log_probs, self.values = [], []
        for _ in range(self.train_interval):
            # obs = obs.transpose(0, 3, 1, 2)
            obs_buffer.append(obs)
            action = self.select_action(obs)
            obs, reward, terminal, info = env.step(action)
            reward = np.clip(reward, -1, 1)
            action_buffer.append(action)
            reward_buffer.append(reward)
            terminal_buffer.append(terminal)

        obs_buffer = np.asarray(obs_buffer, dtype=np.float32)
        reward_buffer = np.asarray(reward_buffer, dtype=np.float32)
        action_buffer = np.asarray(action_buffer, dtype=np.int32)
        terminal_buffer = np.asarray(terminal_buffer, dtype=np.bool)
        self.log_probs = np.asarray(self.log_probs, dtype=np.float32)
        self.values = np.asarray(self.values, dtype=np.float32)

        most_recent_obs = torch.tensor(obs, dtype=torch.float32, device=self.torch_device)
        # most_recent_obs = torch.tensor(obs.transpose(0, 3, 1, 2), dtype=torch.float32, device=self.torch_device)
        _, most_recent_value = self.net(most_recent_obs)
        most_recent_value = most_recent_value.view(-1).cpu().detach().numpy().squeeze()

        # discount_reward_buffer = self.compute_discount_rewards(reward_buffer, terminal_buffer, self.values, most_recent_value)
        
        advantage_buffer = np.zeros_like(reward_buffer)
        last_gae_lam = 0
        for step in reversed(range(self.train_interval)):
            if step == self.train_interval - 1:
                next_value = most_recent_value
            else:
                next_value = self.values[step + 1] 
            next_non_terminal = 1 - terminal_buffer[step] 
            delta = reward_buffer[step] + self.gamma * next_non_terminal * next_value - self.values[step]
            advantage_buffer[step] = last_gae_lam = delta + self.gamma * last_gae_lam * next_non_terminal * self.lam
        discount_reward_buffer = advantage_buffer + self.values

        obs_buffer, action_buffer, discount_reward_buffer, self.values, self.log_probs = \
            map(self._swap_and_flatten, (obs_buffer, action_buffer, discount_reward_buffer, self.values, self.log_probs))
        return obs, obs_buffer, action_buffer, discount_reward_buffer

    def _swap_and_flatten(self, array):
        """swap the axes flatten the array"""
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


        
        