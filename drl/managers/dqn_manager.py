import gym

from drl.managers.base import Manager


class DQNManager(Manager):

    def __init__(self, env, agent):
        Manager.__init__(**locals())
        self.env = env
        from drl.agents.dqn import DQNAgent
        self.agent = DQNAgent()
        from drl.core.memory import ReplayBuffer
        self.memory = ReplayBuffer

    def run(self, episodes):
        raise NotImplementedError

    def save(self):
        raise NotImplementedError

    def load(self):
        raise NotImplementedError

    def save_agent(self):
        raise NotImplementedError

    def load_agent(self):
        raise NotImplementedError

    def save_memory(self):
        raise NotImplementedError

    def load_memory(self):
        raise NotImplementedError

    def close(self):
        raise NotImplementedError


if __name__ == "__main__":

    env = gym.make("CartPole-v0")
    experiment_config = {}
    manager = DQNManager(env, experiment_config)
    manager.run(episodes=100)
    manager.test(episodes=1, render=1)
    manager.save_agent()
    manager.close()
