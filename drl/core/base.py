from abc import ABC, abstractmethod


#a base agent to remove the massive clutter of repeated variable
#assignments
class BaseAgent(ABC):

    #This is needed since locals() also passes self.
    def __init__(self, *args, **kwargs):
        """ Here we reduce the clutter for every agent, storing things hyperparameteres"""
        self._observation_space = kwargs['observation_space']
        self._action_space = kwargs['action_space']
        self._seed = kwargs['seed']
        self._lr = kwargs['lr']
        self._gamma = kwargs['gamma']
        self._batch_size = kwargs['batch_size']

        if self._seed:
            from drl.tools.misc_util import set_seeds
            set_seeds(self._seed)

        #TODO:OpenAI baselines has helpers for the observation inputs..
        # this time we go ham on the class, but this could be made automatically
        #here

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