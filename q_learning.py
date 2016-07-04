import numpy as np

from agent import Agent


class QLearningAgent(Agent):
    def __init__(self, state_space, action_space, discount_factor, greed_eps, learning_params):
        Agent.__init__(self, action_space=action_space)
        self.state_space = state_space
        self.state_count = len(state_space)
        self.discount_factor = discount_factor
        self.greed_eps = greed_eps
        self.learning_params = learning_params

        self.q = np.random.rand(self.state_count, self.action_count) * 0.01

    def perceive(self, state, reward, reached_goal_state=False):
        # perceive
        sp_id = self.state_space.index(state)
        if reached_goal_state:
            self.q[sp_id, :] = reward

        if self.last_action is not None:
            a_id = self.action_space.index(self.last_action)
            s_id = self.state_space.index(self.last_state)

            estimated_q_sa = self.last_reward + self.discount_factor * (np.max(self.q[sp_id, :]))
            self.q[s_id, a_id] += (self.learning_params['LEARNING_RATE'] * (estimated_q_sa - self.q[s_id, a_id]))

        # act
        if np.random.rand() > self.greed_eps:
            s_id = self.state_space.index(state)
            action = self.action_space[np.argmax(self.q[s_id, :])]
        else:
            action = np.random.choice(self.action_space)

        self.last_reward = reward
        self.last_state = state
        self.last_action = action
        return action


