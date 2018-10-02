
import tensorflow as tf
import numpy as np
from tensorflow.distributions import Categorical


slim = tf.contrib.slim

class A2CAgent(object):

  def __init__(self,
               sess, 
               num_actions,
               gamma=0.99,
               tf_device='/cpu:*',
               max_tf_checkpoints_to_keep=3,
               optimizer=tf.train.RMSPropOptimizer(
                 learning_rate=0.00025,
                 decay=0.95,
                 momentum=0.,
                 epsilon=0.00001,
                 centered=True
               )):

    tf.logging.info('Creating %s agent with the following parameters:',
                        self.__class__.__name__)
    tf.logging.info('\t gamma: %f', gamma)
    tf.logging.info('\t tf_device: %s', tf_device)
    tf.logging.info('\t optimzer: %s', optimizer)

    self.num_actions = num_actions
    self.gamma = gamma
    self.optimizer = optimizer

    self.eval_mode = False
    self.step_number = 0
    self.observation_buffer = []
    self.action_buffer = []
    self.reward_buffer = []
    self.terminal_buffer = []
    self.value_buffer = []

    with tf.device(tf_device):
      self.observation_ph = tf.placeholder(tf.float32, [None, 4], 'observation_ph')
      self.action_ph = tf.placeholder(tf.int32, [None], 'action_ph')
      self.target_v_ph = tf.placeholder(tf.float32, [None], 'reward_ph')
      self.advantage_ph = tf.placeholder(tf.float32, [None], 'advantage_ph')

      self._build_network()
      self._train_op = self._build_train_op()

    self.sess = sess


  def _actor_net(self, observation):
    net = slim.fully_connected(observation, 128)
    logits = slim.fully_connected(net, self.num_actions)
    return logits

  def _critic_net(self, observation):
    net = slim.fully_connected(observation, 128)
    v = slim.fully_connected(net, 1)
    return v

  def _build_network(self):
    self.action_logits = self._actor_net(self.observation_ph)
    self.v = self._critic_net(self.observation_ph)

    self.policy_distribution = Categorical(logits=self.action_logits)
    self.choose_actions = self.policy_distribution.sample(1)

  def _build_train_op(self):
    log_probs = self.policy_distribution.log_prob(self.action_ph)
    self.entropy = tf.reduce_mean(self.policy_distribution.entropy())
    self.pg_loss = tf.reduce_mean(self.advantage_ph * (-log_probs))
    self.v_loss = tf.square(self.target_v_ph - tf.squeeze(self.v))

    loss = self.pg_loss - 0.01 * self.entropy + 0.25 * self.v_loss

    return self.optimizer.minimize(loss)
  
  def _select_action(self, observation):
    action = self.sess.run(self.choose_actions, {self.observation_ph: observation})[0]
    if self.eval_mode == True:
      action = action.item()
    return action
    
  def begin_episode(self, observation):
    self.observation_buffer.append(observation)
    action = self._select_action(observation)
    self.action_buffer.append(action)

    value = self.sess.run(self.v, {self.observation_ph: observation}).reshape(-1)
    self.value_buffer.append(value)
    return action

  def step(self, reward, observation, is_terminal):
    if self.eval_mode == True:
      action = self._select_action(observation)
    else:
      self.observation = observation
      self.reward_buffer.append(reward)
      self.terminal_buffer.append(is_terminal)
      self.step_number += 1
      if self.step_number % 5 == 0:
        self._train_step(observation)
        del self.observation_buffer[:]
        del self.action_buffer[:]
        del self.reward_buffer[:]
        del self.value_buffer[:]
        del self.terminal_buffer[:]

      action = self._select_action(observation)
      self.observation_buffer.append(observation)
      self.action_buffer.append(action)

      value = self.sess.run(self.v, {self.observation_ph: observation}).reshape(-1)
      self.value_buffer.append(value)
    return action

  def _train_step(self, observation):
    batch_observation = np.asarray(self.observation_buffer).swapaxes(0, 1).reshape(-1, 4)
    batch_reward = np.asarray(self.reward_buffer, dtype=np.float32).swapaxes(0, 1)
    batch_action = np.asarray(self.action_buffer, dtype=np.int32).swapaxes(0, 1)
    batch_value = np.asarray(self.value_buffer, dtype=np.float32).swapaxes(0, 1)
    batch_terminal = np.asarray(self.terminal_buffer, dtype=bool).swapaxes(0, 1)

    last_value = self.sess.run(self.v, {self.observation_ph: observation}).reshape(-1)

    batch_target_v = []
    for n, (reward, terminal, value) in enumerate(zip(batch_reward, batch_terminal, last_value)):
      target_v = []
      R = value
      for r, t in zip(reward[::-1], terminal[::-1]):
        R = r + self.gamma * R * (1 - t)
        target_v.insert(0, R)
      batch_target_v.append(target_v)
    batch_target_v = np.asarray(batch_target_v)

    batch_target_v = batch_target_v.reshape(-1)
    # batch_terminal = batch_terminal.reshape(-1)
    batch_action = batch_action.reshape(-1)
    batch_value = batch_value.reshape(-1)

    batch_advantage = batch_target_v - batch_value


    self.sess.run(self._train_op, {self.observation_ph: batch_observation,
                                   self.action_ph: batch_action,
                                   self.target_v_ph: batch_target_v,
                                   self.advantage_ph: batch_advantage})









    
