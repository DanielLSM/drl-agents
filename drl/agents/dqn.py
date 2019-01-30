import math
import random

import numpy as np

from drl.core.memory import ReplayBuffer
from drl.core.base import BaseAgent
from drl.core.dqn_models import *

from drl.tools.tf_util import get_session
from drl.tools.misc_util import set_seeds


class DQNAgent(BaseAgent):

    #here we build the notes of our DQN agent
    def __init__(self,
                 seed=None,
                 observation_space=None,
                 action_space=None,
                 lr=5e-4,
                 gamma=1.0,
                 **kwargs):

        sess = get_session()
        set_seeds(seed)

    def act(self, observation):
        """ Agent acts by delivering an action from an observation """
        pass

    def train(self, batch_training=False):
        """ Train the agent according a batch or step """
        pass


if __name__ == '__main__':

    agent = DQNAgent()