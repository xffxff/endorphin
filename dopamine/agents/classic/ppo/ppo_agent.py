
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
                 train_period=128,
                 v_loss_coef=0.25,
                 entropy_coef=0.01,
                 torch_device='cpu'):
        
        self.num_actions = num_actions
        self.gamma = gamma
        self.train_period = train_period
        self.v_loss_coef = v_loss_coef
        self.entropy_coef = entropy_coef
        self.torch_device = torch_device

        self.net = Net(num_actions)

        self.step_num = 0
        self.obs_buffer = []
        self.action_buffer = []
        self.reward_buffer = []
        self.terminal_buffer = []
        

    def _select_action(self, obs):
        """Select an action from the set of available actions.

        Returns:
            int, the selected action.
        """
        logits, value = self.net(obs)
        m = Categorical(logits=logits)
        action = m.sample()
        return action

    def begin_episode(self, obs):
        """Return the agent's first action for this episode.

        Args:
            obs: numpy.ndarray, the environment's initial observation.
        
        Returns:
            int, the selected action.
        """
        self.obs_buffer.append(obs)
        action = self._select_action(obs)
        self.action_buffer.append(action)
        return action
        
    def step(self, reward, obs, terminal):
        """Records the most recent transition and returns the agent's next action.

        Args:
            reward: numpy.ndarray, reward recieved from the agent's most recent action.
            obs: numpy.ndarray, the most recent observation.
            terminal: numpy.ndarray, whether the agent get the terminal state.

        Returns:
            int, the selected action. 
        """
        self.reward_buffer.append(reward)
        self.terminal_buffer.append(terminal)

        self.step_num += 1

        if self.step_num % self.train_period == 0:
            self._train(obs)
        self.obs_buffer.append(obs)
        action = self._select_action(obs)
        self.action_buffer.append(action)

    def _train(self, obs):
        """Train every regular steps.

        Args:
            obs: numpy.ndarray, the most recent observation the agent recieved.
        """
