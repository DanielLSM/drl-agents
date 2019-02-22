import time
import gym

from drl.tools.misc_util import LinearSchedule
from drl.agents.dqn import DQNAgent
from drl.managers.base import Manager
from drl.core.memory import ReplayBuffer
from drl.tools.plotter_util import Plotter

from schedule_sim.envs.task_day import TaskDay


class DQNManager(Manager):

    def __init__(self,
                 env,
                 seed=None,
                 lr=5e-4,
                 gamma=1.0,
                 buffer_size=50000,
                 total_timesteps=1000000,
                 exploration_fraction=0.1,
                 final_epsilon=0.02,
                 learning_starts=1000,
                 train_freq=1,
                 batch_size=32,
                 target_network_update_freq=500,
                 max_steps_per_episode=10000,
                 render_freq=100,
                 **kwargs):
        Manager.__init__(**locals())
        self.env = env
        self.obs_space = env.observation_space
        self.action_space = env.action_space

        self.agent = DQNAgent(self.obs_space, self.action_space)
        self.memory = ReplayBuffer(self.buffer_size)
        self.plotter = Plotter(
            num_lines=1,
            title=env.spec,
            x_label='episodes',
            y_label='total_reward',
            smooth=True)
        #TODO: this should actually by smarter...
        # exploring linearly based on timesteps is really
        # not good, it should be related to the entropy...
        # for now ill do step exploration decrease but
        # in the future it shoudl be like...... by episode maybe
        self.exploration = LinearSchedule(
            schedule_timesteps=int(
                self.exploration_fraction * self.total_timesteps),
            initial_p=self.epsilon,
            final_p=self.final_epsilon)

    def run(self, episodes=1, render=False):
        t0 = time.time()
        for _ in range(episodes):
            print_render = not bool(_ % self.render_freq)
            t1 = time.time()
            total_reward, steps, info = self._rollout(render=print_render)
            # total_reward, steps = self._rollout(render=False)
            if print_render:
                # import ipdb
                # ipdb.set_trace()
                for key, value in info.items():
                    print("task: {} performed: {}".format(key, value))
            self._pprint_episode(_, steps, total_reward, t1, t0)
            self.plotter.add_points(_, total_reward)
            self.plotter.show()

    def _rollout(self, render=False):
        info = {}
        for _ in range(self.action_space.n):
            info[_] = 0
        obs = self.env.reset()
        total_reward = 0.
        steps = 0
        for _ in range(self.max_steps_per_episode):
            self.epsilon = self.exploration.value(self.total_steps)
            argmax_q_values, action, new_epsilon = self.agent.act(
                obs, new_epsilon=self.epsilon)
            # print("argmax {}, action{}".format(argmax_q_values, action))
            next_obs, reward, done, _info = self.env.step(action[0])
            info[action[0]] += 1
            self.memory.add(obs, action[0], reward, next_obs, float(done))
            obs = next_obs
            self._render_train_update(render)
            steps += 1
            self.total_steps += 1
            total_reward += reward
            if done:
                break
        return total_reward, steps, info

    def _is_training(self):
        return self.total_steps > self.learning_starts and \
                self.total_steps % self.train_freq == 0

    def _is_updating_nets(self):
        return self.total_steps > self.learning_starts and \
                self.total_steps % self.target_network_update_freq == 0

    def _render_train_update(self, render=False):
        if render:
            self.env.render()
        if self._is_training():
            self._train_agent()
        if self._is_updating_nets():
            self.agent.update_target_nets()

    def _train_agent(self):
        batch = self.memory.sample(self.batch_size)
        output = self.agent.train(batch)

    def _pprint_episode(self, episode, steps, total_reward, t1, t0):
        tt = time.time()
        print(
            "episode: {} finished in steps {} \ntotal reward: {}  episode took {}"
            .format(episode, steps, total_reward, tt - t1))
        print("total elapsed time across episodes {}".format(tt - t0))
        print("% used for exploration {}".format(100 * self.epsilon))

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

    # env = gym.make("CartPole-v0")
    parameteres_default_file = "/home/daniel/local-dev/schedule-sim/schedule_sim/envs/config/task_day_custom.yaml"
    render_file = "/home/daniel/local-dev/schedule-sim/schedule_sim/envs/config/render_options.yaml"
    env1 = TaskDay(
        parameters_file=parameteres_default_file,
        reward_scale=10,
        debug=1,
        rendering=True,
        render_file=render_file)
    experiment_config = {}
    # import ipdb
    # # ipdb.set_trace()
    obs1 = env1.reset()
    # obs2 = env.reset()
    # import ipdb
    # ipdb.set_trace()
    manager = DQNManager(env1, experiment_config)
    manager.run(episodes=5000)
    # manager.test(episodes=1, render=1)
    # manager.save_agent()
    # manager.close()
