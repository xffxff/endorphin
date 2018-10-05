

import gym
from stable_baselines.common.vec_env import SubprocVecEnv
import numpy as np
import tensorflow as tf
import time
import sys

from dopamine.common import iteration_statistics

def create_envirtonment(game_name, n_cpu=1):
    if n_cpu > 1:
        env = SubprocVecEnv([lambda: gym.make(game_name) for i in range(n_cpu)])
    else:
        env = gym.make(game_name)
    return env


class Runner(object):
    
    def __init__(self,
                 create_agent_fn,
                 create_environmen_fn=create_envirtonment,
                 game_name='CartPole-v0',
                 num_iterations=100,
                 training_steps=10000,
                 evaluation_steps=5000,
                 max_steps_per_episode=500,
                 ):
        assert create_agent_fn is not None
        self.num_iterations = num_iterations
        self.training_steps = training_steps
        self.evaluation_steps = evaluation_steps
        self.max_steps_per_episode = max_steps_per_episode
        
        self.train_environment = create_environmen_fn(game_name, 4)
        self.eval_environment = create_environmen_fn(game_name)
        
        self.sess = tf.Session('', config=tf.ConfigProto(allow_soft_placement=True))
        self.agent = create_agent_fn(self.sess, self.train_environment)
        
        self.sess.run(tf.global_variables_initializer())

    def _initialize_episode(self):
        """Initialization for a new episode.
  
        Returns:
          action: int, the initial action chosen by the agent.
        """
        if self.agent.eval_mode == True:
            initial_observation = self.eval_environment.reset()
            initial_observation = np.asarray(initial_observation)[None, :]
        else:
            initial_observation = self.train_environment.reset()
        return self.agent.begin_episode(initial_observation)

    def _run_one_step(self, action):
        """Executes a single step in the environment.
  
        Args:
          action: int, the action to perform in the environment.
  
        Returns:
          The observation, reward, and is_terminal values returned from the
            environment.
        """
        if self.agent.eval_mode == True:
            observation, reward, is_terminal, _ = self.eval_environment.step(action)
            observation = np.asarray(observation)[None, :]
        else:
            observation, reward, is_terminal, _ = self.train_environment.step(action)
        return observation, reward, is_terminal

    def _run_one_episode(self):
        """Executes a full trajectory of the agent interacting with the environment.
  
        Returns:
          The number of steps taken and the total reward.
        """
        step_number = 0
        total_reward = 0.

        action = self._initialize_episode()
        is_terminal = False

        # Keep interacting until we reach a terminal state.
        while True:
            if is_terminal or step_number == self.max_steps_per_episode:
                break

            # self._environment.render('human')
            observation, reward, is_terminal = self._run_one_step(action)

            total_reward += reward
            step_number += 1

            # Perform reward clipping.
            reward = np.clip(reward, -1, 1)
            action = self.agent.step(reward, observation, is_terminal)
        return step_number, total_reward

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
            # We use sys.stdout.write instead of tf.logging so as to flush frequently
            # without generating a line break.
            sys.stdout.write('Steps executed: {} '.format(step_count) +
                             'Episode length: {} '.format(episode_length) +
                             'Return: {}\r'.format(episode_return))
            sys.stdout.flush()
        return step_count, sum_returns, num_episodes

    def _run_one_train_phase(self, min_steps):

        del self.agent.observation_buffer[:]
        del self.agent.action_buffer[:]
        del self.agent.reward_buffer[:]
        del self.agent.value_buffer[:]
        del self.agent.terminal_buffer[:]

        step_count = 0

        action = self._initialize_episode()

        while step_count < min_steps:
            observation, reward, is_terminal = self._run_one_step(action)
            step_count += 4
            reward = np.clip(reward, -1, 1)
            action = self.agent.step(reward, observation, is_terminal)


    def _run_train_phase(self):
        self.agent.eval_mode = False
        start_time = time.time()
        self._run_one_train_phase(self.training_steps)
        time_delta = time.time() - start_time
        tf.logging.info('One training phase cost: %.2f', time_delta)

    def _run_eval_phase(self, statistics):
        self.agent.eval_mode = True
        _, sum_returns, num_episodes = self._run_one_eval_phase(
            self.evaluation_steps, statistics
        )
        average_return = sum_returns / num_episodes if num_episodes > 0 else 0.0
        tf.logging.info('Average undiscounted return per evaluation episode: %.2f',
                        average_return)
        statistics.append({'eval_average_return': average_return})
        return num_episodes, average_return


    def _run_one_iteration(self, iteration):
        statistics = iteration_statistics.IterationStatistics()
        tf.logging.info('Starting iteration %d', iteration)
        self._run_train_phase()
        num_episodes_eval, average_reward_eval = self._run_eval_phase(
            statistics)

    def run_experiment(self):
        tf.logging.info('Beginning training...')
        for iteration in range(self.num_iterations):
            self._run_one_iteration(iteration)


    
        