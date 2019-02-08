import math
import random

import numpy as np
import tensorflow as tf
# from drl.core.memory import ReplayBuffer
from drl.core.base import BaseAgent
# from drl.core.dqn_models import *

from drl.tools.tf_util import get_session, get_placeholder, adjust_shape
from drl.core.dqn_models import q_mlp, q_target_update

# from drl.tools.misc_util import set_seeds


class DQNAgent(BaseAgent):

    def __init__(self,
                 observation_space,
                 action_space,
                 hiddens=[512],
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
        num_actions = self._action_space.n

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
            shape=act_shape, dtype=act_dtype, name="action_input")
        self.reward = tf.placeholder(
            shape=[None], dtype=tf.float32, name="reward_input")

        self.done = tf.placeholder(tf.float32, [None], name="done")
        self.importance_sample_weights = tf.placeholder(
            tf.float32, [None], name="weights")

        # ================================================================
        # Here we construct our action-value function Q
        # this will be an MLP, no CNN needed
        # ================================================================

        self.q_values = q_mlp(
            hiddens,
            self.obs_input_node,
            num_actions,
            scope='action_value_function')

        self.q_mlp_vars = tf.get_collection(
            tf.GraphKeys.GLOBAL_VARIABLES,
            scope=tf.get_variable_scope().name + "/action_value_function")

        self.argmax_q_values = tf.argmax(self.q_values, axis=1)

        # ================================================================
        # Here we construct our target action-value function Q
        # ================================================================

        self.q_values_target = q_mlp(
            hiddens,
            self.obs_input_node,
            num_actions,
            scope='action_value_function_target')

        self.q_mlp_target_vars = tf.get_collection(
            tf.GraphKeys.GLOBAL_VARIABLES,
            scope=tf.get_variable_scope().name +
            "/action_value_function_target")

        # ================================================================
        # Bellman equation
        # old estimate
        # Q_old(s_t,a_t)
        # new estimate
        # Q_new(s_t,a_t) = R(s,a_t) + gamma * max_a(Q(s_{t+1},a_t))
        # Objective is to minimize the squared error of the difference
        # between the old and new estimates
        # the difference also mentioned in the literature as td_error(0)
        # the target q_function has 2 connotations, one is the target in
        # supervised learning, the second is the TD target to update the value
        # function for the old state (The TD target)
        # https://en.wikipedia.org/wiki/Temporal_difference_learning
        # ================================================================

        # old estimate
        # Q_old(s_t,a_t)
        self.q_value_old = tf.reduce_sum(
            self.q_values * tf.one_hot(self.action, num_actions), 1)

        # new estimate
        # Q_new(s_t,a_t) = R(s,a_t) + max_a(Q(s_{t+1},a_t))

        # max_a(Q(s_{t+1},a_t)
        self.q_target_max = tf.reduce_max(self.q_values_target, 1)
        self.q_target_max = (1.0 - self.done) * self.q_target_max
        # Q_new(s_t,a_t) = R(s,a_t) + max_a(Q(s_{t+1},a_t))
        self.q_value_new = self.reward + self._gamma * self.q_target_max

        # td_error TD(0) = Q_old - Q_new
        self.td_error = self.q_value_old - tf.stop_gradient(self.q_value_new)
        self.errors = 0.5 * tf.square(self.td_error)
        # mean squared td_erors = (1/2) * (TD(0))

        #TODO: we could use huber_loss
        # we minimize the mean of these weights, unless weights are assigned
        # to this errors
        self.weighted_error = tf.reduce_mean(
            self.importance_sample_weights * self.errors)

        #TODO: gradient normalization is left as an additional exercise
        optimizer = tf.train.AdamOptimizer(learning_rate=self._lr)
        self.optimize = optimizer.minimize(
            self.weighted_error, var_list=self.q_mlp_vars)

        # ================================================================
        # Pointer update q_mlp_target_vars with q_mlp_vars
        # ================================================================

        self.q_update_target_vars = q_target_update(self.q_mlp_vars,
                                                    self.q_mlp_target_vars)

        # ================================================================
        # Finalize graph and initiate all variables
        # ================================================================

        get_session().graph.finalize()
        get_session().run(tf.initialize_all_variables())

        print("### agent graph finalized and ready to use! ###")

    def act(self, observation):
        """ Agent acts by delivering an action from an observation """
        return get_session().run(
            self.argmax_q_values, feed_dict={self.obs_input_node: observation})

    def train(self, batch, batch_training=False):
        """ Train the agent according a batch or step """
        feed_dict = {
            self.obs_input_node: batch[0],
            self.action: batch[1],
            self.obs_input_node_target_net: batch[2],  #next observation
            self.reward: batch[3],
            self.done: batch[4]
        }
        get_session().run([self.optimize], feed_dict=feed_dict)

    def save(self):
        raise NotImplementedError

    def load(self):
        raise NotImplementedError


if __name__ == '__main__':

    import gym

    env = gym.make('CartPole-v0')
    obs_space = env.observation_space
    action_space = env.action_space
    # import ipdb
    # ipdb.set_trace()
    agent = DQNAgent(obs_space, action_space)
