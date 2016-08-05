import numpy as np


from rllib.environment import Environment
from rllib.agent import Agent
from rllib.space import DiscreteStateSpace, DiscreteActionSpace
from rllib.parameter_schedule import GreedyEpsilonLinearSchedule
from rllib.q_learning import QLearningAgent, QTableLookup, QNeuralNetwork
from rllib.rl import calculate_optimal_q_dp
from rllib.policy_gradient import PolicyGradientAgent


class SimpleGridWorldStateSpace(DiscreteStateSpace):
    def __init__(self):
        states = []
        for i in range(4):
            for j in range(3):
                states.append((i, j))

        DiscreteStateSpace.__init__(self, states)

    def is_goal_state(self, state):
        if state in [(3, 1), (3, 2)]:
            return True
        return False

    def get_reward(self, state):
        if state == (3, 2):
            return 1.0
        elif state == (3, 1):
            return -1.0
        elif state == (1, 1):
            return 0.0
        else:
            return -0.04

    def get_initial_state(self):
        return (0, 0)


class SimpleGridWorldActionSpace(DiscreteActionSpace):
    def __init__(self):
        DiscreteActionSpace.__init__(self, actions=['L', 'R', 'U', 'D'])


class SimpleGridWorldEnvironment(Environment):
    def __init__(self):
        """
        This is the example grid world discussed in Chapter 21 of Artificial Intelligence: A Modern Approach
        """
        Environment.__init__(self, state_space=SimpleGridWorldStateSpace(),
                             action_space=SimpleGridWorldActionSpace())

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
        if self._state_out_of_bounds(new_state):
            new_state = self.current_state

        return new_state

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
        if state in [(1, 1), (3, 2), (3, 1)]:
            return self.state_space.get_reward(state), [], np.array([])

        reward = self.state_space.get_reward(state)
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
        Agent.__init__(self, action_space=SimpleGridWorldActionSpace())

    def perceive(self, new_state, reward, available_actions, reached_goal_state=False, episode_end=False):
        return np.random.choice(self.action_space)


if __name__ == "__main__":
    env = SimpleGridWorldEnvironment()

    q_opt = calculate_optimal_q_dp(env, discount_factor=1.0, eps=1e-9)
    print q_opt

    epoch_count = 20
    episodes_per_epoch = 2000
    eps_schedule = GreedyEpsilonLinearSchedule(start_eps=1.0, end_eps=0.1, no_episodes=epoch_count*episodes_per_epoch,
                                               decrease_period=episodes_per_epoch)
    rewards = np.zeros(epoch_count)

    # q_function = QTableLookup(env.state_space, env.action_space, learning_rate=0.05)
    q_function = QNeuralNetwork([], env.state_space, env.action_space, learning_params={'LEARNING_RATE': 0.01})
    q_learner = QLearningAgent(q_function, env.action_space, discount_factor=1.0,
                               greed_eps=eps_schedule)
    for e in range(epoch_count):
        for i in range(episodes_per_epoch):
            s, a, r = env.run(q_learner, np.inf)
            rewards[e] += np.sum(r)
        rewards[e] /= episodes_per_epoch
        print("Epoch {0:d}| Avg. reward per episode: {1:f}".format(e+1, rewards[e]))

    q_learner.set_learning_mode(False)
    reward = 0.0
    for i in range(1000):
        s, a, r = env.run(q_learner, np.inf)
        reward += np.sum(r)
    print("Avg. reward with greedy policy: {0:f}".format(reward/1000))

    for state in env.state_space:
        print q_function.get_q(state)

    """
    pg_learner = PolicyGradientAgent(env.state_space, env.action_space, learning_rate=0.05, greed_eps=eps_schedule,
                                     update_freq=1, apply_baseline=True, clip_gradients=True,
                                     optimizer='gd')

    for e in range(epoch_count):
        for i in range(episodes_per_epoch):
            s, a, r = env.run(pg_learner, np.inf)
            rewards[e] += np.sum(r)
        rewards[e] /= episodes_per_epoch
        print("Epoch {0:d}| Avg. reward per episode: {1:f}".format(e+1, rewards[e]))

    pg_learner.set_learning_mode(False)
    reward = 0.0
    for i in range(1000):
        s, a, r = env.run(pg_learner, np.inf)
        reward += np.sum(r)
    print("Avg. reward with learned policy: {0:f}".format(reward/1000))

    for s in env.state_space:
        print pg_learner.forward(env.state_space.to_vector(s))
    """
