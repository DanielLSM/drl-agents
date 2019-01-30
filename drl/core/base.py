from abc import ABC, abstractmethod


#a base agent
class BaseAgent(ABC):

    def __init__(self, *args, **kwargs):
        """ Here the nodes are iniated """
        pass

    @abstractmethod
    def act(self, observation):

        raise NotImplementedError

    @abstractmethod
    def train(self, batch_training=False):
        """ Train the agent according to a batch or a sample """
        raise NotImplementedError


class SimpleAgent(BaseAgent):

    def __init__(self):
        pass

    def act(self, obs):
        pass

    def train(self, obs):
        pass


if __name__ == '__main__':

    SA = SimpleAgent()