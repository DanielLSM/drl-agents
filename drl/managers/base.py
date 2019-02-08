from abc import ABC, abstractmethod


class Manager(ABC):

    def __init__(self, *args, **kwargs):
        pass

    @abstractmethod
    def run(self):
        raise NotImplementedError

    @abstractmethod
    def test(self):
        raise NotImplementedError

    @abstractmethod
    def save(self):
        raise NotImplementedError

    @abstractmethod
    def load(self):
        raise NotImplementedError

    @abstractmethod
    def save_agent(self):
        raise NotImplementedError

    @abstractmethod
    def load_agent(self):
        raise NotImplementedError

    @abstractmethod
    def save_memory(self):
        raise NotImplementedError

    @abstractmethod
    def load_memory(self):
        raise NotImplementedError

    @abstractmethod
    def close(self):
        raise NotImplementedError