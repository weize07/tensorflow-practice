from __future__ import absolute_import, division, print_function

import random
import sys
import time
import os
import matplotlib.pyplot as plt

import tensorflow as tf
import tensorflow.contrib.eager as tfe
import gym
import numpy as np
import pygame

from pygame.locals import *
from collections import deque
config = tf.ConfigProto(device_count={"CPU": 4}, # limit to num_cpu_core CPU usage
                inter_op_parallelism_threads = 0,
                intra_op_parallelism_threads = 0,
                log_device_placement=True)
tf.enable_eager_execution(config)

DOWN_EVENT = pygame.event.Event(KEYDOWN, {'scancode': 111, 'mod': 0, 'unicode': '', 'key': 273})
CLICK = 1
HOLDON = 0

# Hyper Parameters for DQN
GAMMA = 0.9 # discount factor for target Q
INITIAL_EPSILON = 0.5 # starting value of epsilon
FINAL_EPSILON = 0.01 # final value of epsilon
REPLAY_SIZE = 10000 # experience replay buffer size
BATCH_SIZE = 32 # size of minibatch


class DQNAgent():
    def __init__(self, engine):
        self.engine = engine
        self.epsilon = INITIAL_EPSILON
        self.replay_buffer = deque()
        self.action_dim = 2
        self.state_dim = 10
        self.create_Q_network()

    def ready(self):
        pygame.event.post(DOWN_EVENT)

    def memorize(self, state, action, reward, next_state, done):
        one_hot_action = np.zeros(self.action_dim)
        one_hot_action[action] = 1
        # print(state)
        # print(one_hot_action)
        # print(reward)
        # print(next_state)
        # print(done)
        # exit(0)
        print(state.shape)
        self.replay_buffer.append((state,one_hot_action,reward,next_state,done))
        if len(self.replay_buffer) > REPLAY_SIZE:
          self.replay_buffer.popleft()

    def create_Q_network(self):
      self.model = tf.keras.Sequential([
        tf.keras.layers.Dense(20, activation="relu", input_shape=(self.state_dim,)),  # input shape required
        tf.keras.layers.Dense(256, activation="relu"),
        tf.keras.layers.Dense(self.action_dim)
      ])
      self.optimizer = tf.train.AdamOptimizer(learning_rate=0.001)

    def loss(self, model, states, actions, q_values):
      tmp_qs = model(states)
      y_ = tf.multiply(actions, tmp_qs)
      y_ = tf.reshape(tf.reduce_sum(y_, 1), [BATCH_SIZE, 1,])

      loss = tf.losses.mean_squared_error(labels=q_values, predictions=y_)
      return loss

    def grad(self, model, states, actions, q_values):
      with tfe.GradientTape() as tape:
        loss_value = self.loss(model, states, actions, q_values)
      return tape.gradient(loss_value, model.variables)

    def train_Q_network(self):
      # Step 1: obtain random minibatch from replay memory
      minibatch = random.sample(self.replay_buffer, BATCH_SIZE)
      # print(minibatch)
      state_batch = [data[0] for data in minibatch]
      action_batch = [data[1] for data in minibatch]
      reward_batch = [data[2] for data in minibatch]
      next_state_batch = [data[3] for data in minibatch]

      # Step 2: calculate y
      Q_value_batch = []
      for i in range(0, BATCH_SIZE):
        # s = tf.convert_to_tensor(next_state_batch[i], dtype=tf.float32)
        s = tf.convert_to_tensor(next_state_batch[i])
        s = tf.reshape(s, [1,self.state_dim])
        Q_value_batch.append(self.model(s))

      y_batch = []
      for i in range(0, BATCH_SIZE):
        done = minibatch[i][4]
        if done:
          y_batch.append(reward_batch[i])
        else :
          y_batch.append(reward_batch[i] + GAMMA * np.max(Q_value_batch[i]))
      # state_tensor = tf.convert_to_tensor(state_batch)
      state_tensor = tf.convert_to_tensor(state_batch, dtype=tf.float32)
      # action_tensor = tf.convert_to_tensor(action_batch)
      action_tensor = tf.convert_to_tensor(action_batch, dtype=tf.float32)
      # reward_tensor = tf.convert_to_tensor(y_batch)
      reward_tensor = tf.convert_to_tensor(y_batch, dtype=tf.float32)
      reward_tensor = tf.reshape(reward_tensor, [BATCH_SIZE, 1])

      # Optimize the model
      grads = self.grad(self.model, state_tensor, action_tensor, reward_tensor)
      self.optimizer.apply_gradients(zip(grads, self.model.variables),
                              global_step=tf.train.get_or_create_global_step())

    def egreedy(self):
        if random.random() <= self.epsilon:
            action = random.randint(HOLDON, CLICK) #
            if action == CLICK:
                pygame.event.post(DOWN_EVENT)
            return action
        else:
            self.epsilon -= (INITIAL_EPSILON - FINAL_EPSILON)/10000
            return self.action()

    def action(self):
        state = self.engine.state()[1:]
        action = self._pie(state)
        if action == CLICK:
            pygame.event.post(DOWN_EVENT)
        return action

    def _pie(self, state):
        s = tf.convert_to_tensor(state, dtype=tf.float32)
        s = tf.reshape(s, [1, self.state_dim])
        return np.argmax(self.model(s))
