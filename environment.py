import numpy as np


class Environment(object):
    def __init__(self, state_space, action_space, goal_states=(), initial_state=None, initial_reward=0.0):
        self.state_space = state_space
        self.state_count = len(state_space)

        self.action_space = action_space
        self.action_count = len(action_space)

        self.goal_states = goal_states

        self.initial_state = initial_state
        self.initial_reward = initial_reward

        self.current_state = None
        self.current_reward = None

    def reset(self):
        self.current_state = self.initial_state
        self.current_reward = self.initial_reward
        if self.initial_state is None:
            self.current_state = np.random.choice(self.state_space)

    def get_next_states(self, state, action):
        pass

    def _advance(self, action):
        raise NotImplementedError()

    def run(self, agent, episode_length):
        self.reset()
        agent.reset()

        # we adopt the convention in Section 21 of Artificial Intelligence by Russell, Norvig
        # s_0, s_1, ..., s_T
        states = []
        # a_t is the action taken at state s_t
        # a_0, a_1, ..., a_T-1, a_T=None
        actions = []
        # r_t is the reward received associated with state s_t
        # r_0, r_2, ..., r_T
        rewards = []

        e = 0
        while True:
            action = agent.perceive(self.current_state, self.current_reward,
                                    reached_goal_state=self.current_state in self.goal_states,
                                    episode_end=False)

            states.append(self.current_state)
            actions.append(action)
            rewards.append(self.current_reward)
            self.current_state, self.current_reward = self._advance(action)

            e += 1
            if e >= episode_length or self.current_state in self.goal_states:
                # let the agent perceive one last time.
                # note that we append None as its action because the agent does not act in the terminal state
                _ = agent.perceive(self.current_state, self.current_reward,
                                   reached_goal_state=self.current_state in self.goal_states,
                                   episode_end=e>=episode_length)
                states.append(self.current_state)
                rewards.append(self.current_reward)
                actions.append(None)
                break

        return np.array(states), np.array(actions), np.array(rewards)

