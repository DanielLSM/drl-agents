import gym

from drl.agents.dqn import DQNAgent
from drl.managers.base import Manager
from drl.core.memory import ReplayBuffer
from drl.tools.plotter_util import Plotter


class DQNManager(Manager):

    def __init__(self,
                 env,
                 agent,
                 seed=None,
                 lr=5e-4,
                 buffer_size=50000,
                 final_epsilon=0.02,
                 train_freq=1,
                 batch_size=32,
                 gamma=1.0,
                 target_network_update_freq=500,
                 max_steps_per_episode=1000):
        Manager.__init__(**locals())
        self.env = env
        obs_space = env.observation_space
        action_space = env.action_space
        self.agent = DQNAgent(obs_space, action_space)
        self.memory = ReplayBuffer(buffer_size)
        self.plotter = Plotter(num_lines=1)
        self.total_steps = 0
        self.epsilon = 1.0
        self.max_steps_per_episode = max_steps_per_episode
        self.train_freq = train_freq

    def run(self, episodes=1):
        import time
        t0 = time.time()
        for _ in range(episodes):
            total_reward, steps = self._rollout(render=False)
            self.plotter.add_points(_, total_reward)
            self.plotter.show()
            print(
                "episode: {} finished in steps {} \n total reward: {}  elapsed time {}"
                .format(_, total_reward, steps,
                        time.time() - t0))

    def _rollout(self, render=False):

        obs = self.env.reset()
        total_reward = 0.

        for _ in range(self.max_steps_per_episode):
            action = self.agent.act(obs)
            next_obs, reward, done, _info = env.step(action)
            self.memory.add(obs, action, reward, next_obs, float(done))
            obs = next_obs
            if render:
                self.env.render()
            if self.total_steps % self.train_freq:
                self._train_agent()
            self.total_steps += 1
            if done:
                break
        return total_reward

    def _train_agent(self):
        pass

    def test(self, episodes=1, render=False):
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
