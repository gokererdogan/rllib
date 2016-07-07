import numpy as np

from agent import Agent


class QLearningAgent(Agent):
    def __init__(self, state_space, action_space, discount_factor, greed_eps, learning_rate):
        Agent.__init__(self, action_space=action_space)
        self.state_space = state_space
        self.state_count = len(state_space)
        self.discount_factor = discount_factor
        self.greed_eps = greed_eps
        self.learning_rate = learning_rate
        self.episodes_experienced = 0

        self.q = np.random.rand(self.state_count, self.action_count) * 0.01

    def perceive(self, state, reward, available_actions, reached_goal_state=False, episode_end=False):
        if self.learning_on:
            # perceive
            sp_id = self.state_space.index(state)
            if reached_goal_state or episode_end:
                self.episodes_experienced += 1
                if reached_goal_state:
                    self.q[sp_id, :] = reward

            if self.last_action is not None:
                a_id = self.action_space.index(self.last_action)
                s_id = self.state_space.index(self.last_state)

                estimated_q_sa = self.last_reward + self.discount_factor * (np.max(self.q[sp_id, :]))
                self.q[s_id, a_id] += (self.learning_rate * (estimated_q_sa - self.q[s_id, a_id]))

        # act
        if reached_goal_state or episode_end:
            action = None
        else:
            if not self.learning_on or np.random.rand() > self.greed_eps.get_value(self):
                available_action_ids = [self.action_space.index(a) for a in available_actions]
                s_id = self.state_space.index(state)
                # mask unavailable actions
                q_s = self.q[s_id, :]
                mask_a = [i not in available_action_ids for i in range(len(q_s))]
                q_ma = np.ma.masked_array(self.q[s_id, :], mask=mask_a)
                action = self.action_space[q_ma.argmax()]
            else:
                a_id = np.random.choice(len(available_actions))
                action = available_actions[a_id]

        self.last_reward = reward
        self.last_state = state
        self.last_action = action
        return action


