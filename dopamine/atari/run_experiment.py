
import gym
import time
import sys
import numpy as np
from dopamine.atari.preprocessing import AtariPreprocessing
from stable_baselines.common.vec_env import SubprocVecEnv



def create_atari_environment(game_name, stick_actions=True):
    game_version = 'v0' if stick_actions else 'v4'
    full_game_name = '{}NoFrameskip-{}'.format(game_name, game_version)
    env = gym.make(full_game_name)

    env = env.env
    env = AtariPreprocessing(env)
    return env

def create_multi_environment(env, n_cpu):
    multi_env = SubprocVecEnv([lambda: env for i in range(n_cpu)])
    return multi_env


class Runner(object):

    def __init__(self,
                 create_agent_fn,
                 game_name='CartPole-v0',
                 stick_actions=True,
                 num_iters=200,
                 train_steps=10000,
                 eval_steps=5000,
                 max_steps_per_episode=27000):
        self.num_iters = num_iters
        self.train_steps = train_steps
        self.eval_steps = eval_steps
        self.max_steps_per_episode = max_steps_per_episode

        self.eval_env = gym.make(game_name)
        # self.eval_env = create_atari_environment(game_name, stick_actions)
        self.train_env = create_multi_environment(self.eval_env, 4)
        self.env = self.train_env

        self.agent = create_agent_fn(self.env)

    def _initialize_episode(self):
        initial_observation = self.env.reset()
        return self.agent.begin_episode(initial_observation)

    def _run_one_step(self, action):
        observation, reward, is_terminal, _ = self.env.step(action)
        return observation, reward, is_terminal

    def _run_one_episode(self):
        step_num = 0
        total_reward = 0.

        action = self._initialize_episode()
        is_terminal = False

        while True:
            observation, reward, is_terminal = self._run_one_step(action)
            total_reward += reward
            step_num += 1

            reward = np.clip(reward, -1, 1)

            if is_terminal or step_num == self.max_steps_per_episode:
                break
            action = self.agent.step(reward, observation, is_terminal)

        return step_num, total_reward

    def _run_one_eval_phase(self, min_steps):
        step_count = 0
        num_episodes = 0
        sum_returns = 0. 

        while step_count < min_steps:
            episode_length, episode_return = self._run_one_episode()

            step_count += episode_length
            sum_returns += episode_return
            num_episodes += 1

            sys.stdout.write('Steps executed: {} '.format(step_count) +
                             'Episode length: {} '.format(episode_length) +
                             'Return: {}\r'.format(episode_return))
            sys.stdout.flush()

        return step_count, sum_returns, num_episodes

    def _run_eval_phase(self):
        self.env = self.eval_env
        self.agent.eval_mode = True
        _, sum_returns, num_episodes = self._run_one_eval_phase(
            self.eval_steps)
        average_return = sum_returns / num_episodes if num_episodes > 0 else 0.0

        print('Average undiscounted return per evaluation episode: %.2f', 
              average_return)
        
        return num_episodes, average_return

    def _run_one_train_phase(self, min_steps):
        del self.agent.obs_buffer[:]
        del self.agent.action_buffer[:]
        del self.agent.reward_buffer[:]
        # del self.agent.value_buffer[:]
        del self.agent.terminal_buffer[:]

        step_count = 0

        action = self._initialize_episode()

        while step_count < min_steps:
            observation, reward, terminal = self._run_one_step(action)
            step_count += 4
            # print(step_count)
            reward = np.clip(reward, -1, 1)
            action = self.agent.step(reward, observation, terminal)

    def _run_train_phase(self):
        self.env = self.train_env
        self.agent.eval_mode = False
        start_time = time.time()
        self._run_one_train_phase(self.train_steps)
        time_delta = time.time() - start_time
        print('One training phase cost: %.2f', time_delta)

    def _run_one_iteration(self, iteration):
        print('Starting iteration %d', iteration)
        self._run_train_phase()
        num_episodes_eval, average_reward_eval = self._run_eval_phase()

    def run_experiment(self):
        print('Beginning training...')
        for iteration in range(self.num_iters):
            self._run_one_iteration(iteration)
