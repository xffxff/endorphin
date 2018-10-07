
import os
import gym
import time
import sys
import numpy as np
from dopamine.atari import preprocessing
from dopamine.common import iteration_statistics
from dopamine.common import logger
from dopamine.common import checkpointer
from stable_baselines.common.vec_env import SubprocVecEnv
from stable_baselines.common.cmd_util import make_atari_env
from stable_baselines.common.vec_env.vec_frame_stack import VecFrameStack


def create_atari_environment(game_name, sticky_actions=True, n_cpu=1):
    game_version = 'v0' if sticky_actions else 'v4'
    full_game_name = '{}NoFrameskip-{}'.format(game_name, game_version)
    env = gym.make(full_game_name)
    # Strip out the TimeLimit wrapper from Gym, which caps us at 100k frames. We
    # handle this time limit internally instead, which lets us cap at 108k frames
    # (30 minutes). The TimeLimit wrapper also plays poorly with saving and
    # restoring states.
    env = env.env
    env = preprocessing.AtariPreprocessing(env)
    env = preprocessing.FrameStackPreprocessing(env)

    # env = make_atari_env(full_game_name, n_cpu, seed=123)
    # env = VecFrameStack(env, 4)

    return env

def create_multi_environment(env, n_cpu):
    multi_env = SubprocVecEnv([lambda: env for i in range(n_cpu)])
    return multi_env



class Runner(object):

    def __init__(self,
                 create_agent_fn,
                 base_dir,
                 game_name='Pong',
                 sticky_actions=True,
                 n_cpu = 4,
                 num_iters=200,
                 train_steps=10000,
                 eval_steps=5000,
                 log_every_n=1,
                 log_file_prefix='log',
                 checkpoint_file_prefix='ckpt',
                 max_steps_per_episode=27000):
        self.base_dir = base_dir
        self.num_iters = num_iters
        self.n_cpu = n_cpu
        self.train_steps = train_steps
        self.eval_steps = eval_steps
        self.log_every_n = log_every_n
        self.log_file_prefix = log_file_prefix
        self.max_steps_per_episode = max_steps_per_episode

        # self.eval_env = create_atari_environment(game_name, sticky_actions=sticky_actions, n_cpu=1)
        # self.train_env = create_atari_environment(game_name, sticky_actions=sticky_actions, n_cpu=n_cpu)

        self.eval_env = create_atari_environment(game_name, sticky_actions=sticky_actions)
        self.train_env = create_multi_environment(self.eval_env, n_cpu)
        self.env = self.train_env

        self.agent = create_agent_fn(self.env)
        self._create_directories()  
        self._initialize_checkpointer_and_maybe_resume(checkpoint_file_prefix)

    def _create_directories(self):
        self.checkpoint_dir = os.path.join(self.base_dir, 'checkpoints')
        self.logger = logger.Logger(os.path.join(self.base_dir, 'logs'))

    def _initialize_checkpointer_and_maybe_resume(self, checkpoint_file_prefix):
        self.checkpointer = checkpointer.Checkpointer(self.checkpoint_dir, checkpoint_file_prefix)
        self.start_iteration = 0

        latest_checkpoint_version = checkpointer.get_latest_checkpoint_number(self.checkpoint_dir)
        if latest_checkpoint_version >= 0:
            experiment_data = self.checkpointer.load_checkpoint(latest_checkpoint_version)
            if self.agent.unbundle(
                self.checkpoint_dir, latest_checkpoint_version, experiment_data):
                assert 'logs' in experiment_data
                assert 'current_iteration' in experiment_data
                self.logger.data = experiment_data['logs']
                self.start_iteration = experiment_data['current_iteration'] + 1
                print('Reloaded checkpoint and will start from iteration ', self.start_iteration)

    def _initialize_episode(self):
        initial_observation = self.env.reset()
        # initial_observation = np.transpose(initial_observation, (0, 3, 1, 2))
        return self.agent.begin_episode(initial_observation)

    def _run_one_step(self, action):
        observation, reward, is_terminal, _ = self.env.step(action)
        # observation = np.transpose(observation, (0, 3, 1, 2))
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

    def _run_one_eval_phase(self, min_steps, statistics):
        step_count = 0
        num_episodes = 0
        sum_returns = 0. 

        while step_count < min_steps:
            episode_length, episode_return = self._run_one_episode()

            statistics.append({
                '{}_episode_lengths'.format('eval'): episode_length,
                '{}_episode_returns'.format('eval'): episode_return
            })
            step_count += episode_length
            sum_returns += episode_return
            num_episodes += 1

            sys.stdout.write('Steps executed: {} '.format(step_count) +
                             'Episode length: {} '.format(episode_length) +
                             'Return: {}\r'.format(episode_return))
            sys.stdout.flush()

        return step_count, sum_returns, num_episodes

    def _run_eval_phase(self, statistics):
        self.env = self.eval_env
        self.agent.eval_mode = True
        _, sum_returns, num_episodes = self._run_one_eval_phase(
            self.eval_steps, statistics)
        average_return = sum_returns / num_episodes if num_episodes > 0 else 0.0

        print('Average undiscounted return per evaluation episode: ', 
              average_return)
        statistics.append({'eval_average_return': average_return})

    def _run_one_train_phase(self, min_steps):
        del self.agent.obs_buffer[:]
        del self.agent.action_buffer[:]
        del self.agent.reward_buffer[:]
        del self.agent.terminal_buffer[:]

        step_count = 0

        action = self._initialize_episode()

        while step_count < min_steps:
            observation, reward, terminal = self._run_one_step(action)
            step_count += self.n_cpu
            reward = np.clip(reward, -1, 1)
            action = self.agent.step(reward, observation, terminal)

    def _run_train_phase(self):
        self.env = self.train_env
        self.agent.eval_mode = False
        start_time = time.time()
        self._run_one_train_phase(self.train_steps)
        time_delta = time.time() - start_time
        print('One training phase cost: ', time_delta, 's')

    def _run_one_iteration(self, iteration):
        statistics = iteration_statistics.IterationStatistics()
        print('Starting iteration ', iteration)
        self._run_train_phase()
        self._run_eval_phase(statistics)
        return statistics.data_lists

    def _log_experiment(self, iteration, statistics):
        self.logger['iteration_{:d}'.format(iteration)] = statistics
        if iteration % self.log_every_n == 0:
            self.logger.log_to_file(self.log_file_prefix, iteration)

    def _checkpoint_experiment(self, iteration):
        experiment_data = self.agent.bundle_and_checkpoint(self.checkpoint_dir, iteration)
        
        # if experiment_data:
        experiment_data['current_iteration'] = iteration
        experiment_data['logs'] = self.logger.data
        self.checkpointer.save_checkpoint(iteration, experiment_data)

    def run_experiment(self):
        print('Beginning training...')
        for iteration in range(self.num_iters):
            statistics = self._run_one_iteration(iteration)
            # self._log_experiment(iteration, statistics)
            # self._checkpoint_experiment(iteration)
