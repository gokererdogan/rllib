class Agent(object):
    def __init__(self, action_space):
        self.action_space = action_space
        self.action_count = len(action_space)

        self.last_state = None
        self.last_action = None
        self.last_reward = 0.0

    def reset(self):
        self.last_state = None
        self.last_action = None
        self.last_reward = 0.0

    def get_action_probabilities(self, state):
        pass

    def perceive(self, state, reward, reached_goal_state=False, episode_end=False):
        raise NotImplementedError()


