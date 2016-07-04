import numpy as np
import scipy.stats as spst

from gmllib.helpers import progress_bar

from workspace.rl.environment import Environment
from workspace.rl.q_learning import QLearningAgent


class RealLineEnvironment(Environment):
    def __init__(self, start, end, state_count, reward_func):
        Environment.__init__(self, state_space=list(np.linspace(start, end, state_count)),
                             action_space=['L', 'R'])
        self.reward_function = reward_func

    def reset(self):
        Environment.reset(self)
        self.initial_reward = self.reward_function(self.current_state)

    def _advance(self, action):
        s_id = self.state_space.index(self.current_state)
        if s_id == 0 and action == 'L':
            return self.current_state, self.reward_function(self.current_state)
            # return self.current_state, 0.0
        elif s_id == self.state_count - 1 and action == 'R':
            return self.current_state, self.reward_function(self.current_state)
            # return self.current_state, 0.0
        else:
            if action == 'L':
                new_state = self.state_space[s_id - 1]
            elif action == 'R':
                new_state = self.state_space[s_id + 1]
            return new_state, self.reward_function(new_state)

"""
class RealLineMHEnvironment(MHEnvironment):
    def __init__(self, start, end, state_count, state_probability, agent):
        MHEnvironment.__init__(self, state_space=np.linspace(start, end, state_count),
                               state_probability=state_probability, agent=agent)

    def get_next_states(self, state, action):
        next_states = [state]
        next_state_probabilities = []

        state_id = np.argwhere(self.state_space == state)[0, 0]
        next_state_id = state_id + action

        if self.state_count > next_state_id >= 0:
            next_state = self.state_space[next_state_id]
            next_states.append(next_state)

            p_next_state = self.state_probability(next_state)
            p_state = self.state_probability(state)

            q_current_next = self.agent.get_proposal_probability(state, next_state)
            q_next_current = self.agent.get_proposal_probability(next_state, state)

            acc_ratio = min((p_next_state * q_next_current) / (p_state * q_current_next), 1.0)

            next_state_probabilities.append(1.0 - acc_ratio)
            next_state_probabilities.append(acc_ratio)

        else:
            next_state_probabilities.append(1.0)

        rewards = []
        for s in next_states:
            rewards.append(self.state_probability(s))

        return next_states, np.array(next_state_probabilities), np.array(rewards)

    def _advance(self, action):
        state = self.current_state
        state_id = np.argwhere(self.state_space == state)[0, 0]

        next_state_id = state_id + action
        if next_state_id < 0 or next_state_id >= self.state_count:
            return state, self.state_probability(state)

        next_state = self.state_space[next_state_id]

        p_state = self.state_probability(state)
        p_next_state = self.state_probability(next_state)

        q_current_next = self.agent.get_proposal_probability(state, next_state)
        q_next_current = self.agent.get_proposal_probability(next_state, state)

        acc_ratio = min((p_next_state * q_next_current) / (p_state * q_current_next), 1.0)

        if np.random.rand() < acc_ratio:
            self.current_state = next_state

        reward = self.state_probability(self.current_state)

        return self.current_state, reward


class LeftRightAgent(MHAgent):
    def __init__(self):
        MHAgent.__init__(self, action_space=np.array([-1, 1]))

    def get_action_probabilities(self, state):
        p_a = np.ones(2) * 0.5
        return p_a

    def act(self, state):
        return np.random.choice(self.action_space)

    def perceive(self, new_state, reward):
        pass

    def get_proposal_probability(self, current_state, proposed_state):
        return 0.5
"""


def p(x):
    return 0.7 * spst.norm.pdf(x, 2.0, .5) + 0.3 * spst.norm.pdf(x, 5.0, .5)


if __name__ == "__main__":
    environment = RealLineEnvironment(0.0, 8.0, 41, p)

    q_learner = QLearningAgent(environment.state_space, ['L', 'R'], discount_factor=0.9,
                               greed_eps=0.1, learning_params={'LEARNING_RATE': 0.1})
    for i in range(1000):
        progress_bar(i, max=1000, update_freq=10)
        environment.run(q_learner, episode_length=500)

    q = q_learner.q
    print q

    import matplotlib.pyplot as plt

    plt.figure()

    x = environment.state_space
    px = np.array([p(x_i) for x_i in x])
    plt.plot(x, px)

    # plt.plot(x, q)
    # plt.legend(["-1", "1"])

    p1 = np.exp(q[:, 1]) / np.sum(np.exp(q), 1)
    # p1[np.isnan(p1)] = 0.5
    plt.plot(x, p1)

    plt.axhline(y=0.5, xmin=np.min(x), xmax=np.max(x))

    plt.show()

