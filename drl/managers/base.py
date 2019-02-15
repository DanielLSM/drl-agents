from abc import ABC, abstractmethod


class Manager(ABC):

    def __init__(self, *args, **kwargs):
        self.seed = kwargs['seed']
        self.lr = kwargs['lr']
        self.gamma = kwargs['gamma']
        self.buffer_size = kwargs['buffer_size']
        self.total_timesteps = kwargs['total_timesteps']
        self.exploration_fraction = kwargs['exploration_fraction']
        self.final_epsilon = kwargs['final_epsilon']
        self.learning_starts = kwargs['learning_starts']
        self.train_freq = kwargs['train_freq']
        self.batch_size = kwargs['batch_size']
        self.max_steps_per_episode = kwargs['max_steps_per_episode']
        self.target_network_update_freq = kwargs['target_network_update_freq']
        self.render_freq = kwargs['render_freq']
        self.total_steps = 0
        self.epsilon = 1.0

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