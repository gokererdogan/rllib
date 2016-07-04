import numpy as np

from rllib.environment import Environment
from rllib.agent import Agent
from rllib.q_learning import QLearningAgent
from rllib.rl import calculate_optimal_q_dp
from rllib.policy_gradient import PolicyGradientAgent


class SimpleGridWorldEnvironment(Environment):
    def __init__(self):
        """
        This is the example grid world discussed in Chapter 21 of Artificial Intelligence: A Modern Approach
        """
        state_space = []
        for i in range(4):
            for j in range(3):
                state_space.append((i, j))

        Environment.__init__(self, state_space=state_space,
                             action_space=['L', 'R', 'U', 'D'],
                             initial_state=(0, 0), initial_reward=-0.04,
                             goal_states=((3, 1), (3, 2)))

        self.action_steps = {'L': (-1, 0), 'R': (1, 0), 'U': (0, 1), 'D': (0, -1)}
        self.relative_actions = {('L', 'L'): 'D', ('L', 'R'): 'U', ('R', 'L'): 'U', ('R', 'R'): 'D',
                                 ('U', 'L'): 'L', ('U', 'R'): 'R', ('D', 'L'): 'R', ('D', 'R'): 'L'}
        self.x_width = 4
        self.y_width = 3

    def _advance(self, action=None):
        if np.random.rand() > 0.8:
            err = np.random.choice(['L', 'R'])
            action = self.relative_actions[(action, err)]

        step = self.action_steps[action]
        new_state = (self.current_state[0] + step[0], self.current_state[1] + step[1])
        if new_state == (3, 2):
            return new_state, 1.0
        elif new_state == (3, 1):
            return new_state, -1.0
        elif self._state_out_of_bounds(new_state):
            return self.current_state, -0.04
        else:
            return new_state, -0.04

    def _state_out_of_bounds(self, state):
        if state[0] < 0 or state[0] >= self.x_width or state[1] < 0 or state[1] >= self.y_width or state == (1, 1):
            return True
        return False

    def _add_next_state(self, next_states, state_probabilities, state, new_state, prob):
        if self._state_out_of_bounds(new_state):
            if state in next_states:
                s_id = next_states.index(state)
                state_probabilities[s_id] += prob
            else:
                next_states.append(state)
                state_probabilities.append(prob)
        else:
            next_states.append(new_state)
            state_probabilities.append(prob)

    def get_next_states(self, state, action):
        if state == (1, 1):
            return 0.0, [], np.array([])
        elif state == (3, 2):
            return 1.0, [], np.array([])
        elif state == (3, 1):
            return -1.0, [], np.array([])

        reward = -0.04
        next_states = []
        state_probabilities = []

        # no err
        step = self.action_steps[action]
        new_state = (state[0] + step[0], state[1] + step[1])
        self._add_next_state(next_states, state_probabilities, state, new_state, 0.8)

        # err left
        step = self.action_steps[self.relative_actions[(action, 'L')]]
        new_state = (state[0] + step[0], state[1] + step[1])
        self._add_next_state(next_states, state_probabilities, state, new_state, 0.1)

        # err right
        step = self.action_steps[self.relative_actions[(action, 'R')]]
        new_state = (state[0] + step[0], state[1] + step[1])
        self._add_next_state(next_states, state_probabilities, state, new_state, 0.1)

        return reward, next_states, np.array(state_probabilities)


class SimpleGridWorldAgent(Agent):
    def __init__(self):
        Agent.__init__(self, action_space=['L', 'R', 'U', 'D'])

    def perceive(self, new_state, reward, reached_goal_state=False, episode_end=False):
        return np.random.choice(self.action_space)


if __name__ == "__main__":
    env = SimpleGridWorldEnvironment()
    action_space = ['L', 'R', 'U', 'D']

    """
    q_opt = calculate_optimal_q_dp(env, action_space, discount_factor=1.0, eps=1e-9)
    print q_opt

    q_learner = QLearningAgent(env.state_space, action_space, discount_factor=1.0,
                               greed_eps=0.1, learning_params={'LEARNING_RATE': 0.05})

    for i in range(50000):
        env.run(q_learner, np.inf)

    print q_learner.q
    """

    pg_learner = PolicyGradientAgent(env.state_space, action_space, learning_rate=0.05,
                                     update_freq=1, optimizer='gd')

    epoch_count = 20
    episodes_per_epoch = 1000
    rewards = np.zeros(epoch_count)
    for e in range(epoch_count):
        for i in range(episodes_per_epoch):
            s, a, r = env.run(pg_learner, np.inf)
            rewards[e] += np.sum(r)
        rewards[e] /= episodes_per_epoch
        print("Epoch {0:d}| Avg. reward per episode: {1:f}".format(e+1, rewards[e]))

