from abc import ABC, abstractmethod


#a base agent
class BaseAgent(ABC):

    def __init__(self, *args, **kwargs):
        pass

    @abstractmethod
    def act(self, observation):

        raise NotImplementedError


class SimpleAgent(BaseAgent):

    def __init__(self):
        pass

    def act(self, obs):
        pass


if __name__ == '__main__':

    SA = SimpleAgent()