import math
import random

import numpy as np
import tensorflow as tf
# from drl.core.memory import ReplayBuffer
from drl.core.base import BaseAgent
# from drl.core.dqn_models import *

from drl.tools.tf_util import get_session, get_placeholder

# from drl.tools.misc_util import set_seeds


class DQNAgent(BaseAgent):

    def __init__(self,
                 observation_space,
                 action_space,
                 seed=None,
                 lr=5e-4,
                 gamma=1.0,
                 batch_size=32,
                 **kwargs):
        """ Setup of agent's variables and graph construction with useful
        pointers to nodes  """
        BaseAgent.__init__(**locals())

        # Declaring for readibility
        batch_size = self._batch_size
        obs_shape = self._observation_space.shape
        obs_dtype = self._observation_space.dtype

        act_shape = self._action_space.shape
        act_dtype = self._action_space.dtype

        # ================================================================
        # Input nodes of the graph, obervations, actions
        # and hyperparameters, aka tf.placeholders
        # ================================================================

        self.obs_input_node = tf.placeholder(
            shape=(batch_size,) + obs_shape,
            dtype=obs_dtype,
            name="observation_input")

        self.obs_input_node_target_net = tf.placeholder(
            shape=(batch_size,) + obs_shape,
            dtype=obs_dtype,
            name="observation_input_target_net")

        self.action = tf.placeholder(
            shape=[None], dtype=tf.int32, name="action_input")
        self.reward = tf.placeholder(
            shape=[None], dtype=tf.float32, name="reward_input")

    def act(self, observation):
        """ Agent acts by delivering an action from an observation """
        pass

    def train(self, batch_training=False):
        """ Train the agent according a batch or step """
        pass


if __name__ == '__main__':

    import gym

    env = gym.make('CartPole-v0')
    obs_space = env.observation_space
    act_space = env.action_space
    agent = DQNAgent(obs_space, act_space)
