import numpy as np


class StateSpace(object):
    def __init__(self):
        pass

    def __len__(self):
        raise NotImplementedError()

    def index(self, state):
        raise NotImplementedError()

    def get_reward(self, state):
        raise NotImplementedError

    def get_initial_state(self):
        raise NotImplementedError

    def is_goal_state(self, state):
        raise NotImplementedError()

    def to_string(self, state):
        return str(state)

    def to_vector(self, state):
        # converts the state to a real vector representation
        # required for function approximators like neural networks
        pass


class DiscreteStateSpace(StateSpace):
    def __init__(self, states):
        StateSpace.__init__(self)
        self.states = states

    def __len__(self):
        return len(self.states)

    def index(self, state):
        return self.states.index(state)

    def __getitem__(self, item):
        return self.states[item]

    def __iter__(self):
        return self.states.__iter__()

    def next(self):
        return self.states.next()

    def get_reward(self, state):
        raise NotImplementedError()

    def get_initial_state(self):
        raise NotImplementedError()

    def is_goal_state(self, state):
        raise NotImplementedError()

    def to_vector(self, state):
        s_id = self.index(state)
        x = np.zeros(len(self))
        x[s_id] = 1.0
        return x


class ActionSpace(object):
    def __init__(self):
        pass

    def __len__(self):
        raise NotImplementedError()

    def index(self, action):
        raise NotImplementedError()

    def to_string(self, action):
        return str(action)

    def to_vector(self, action):
        # converts the action to a real vector representation
        # required for function approximators like neural networks
        pass


class DiscreteActionSpace(ActionSpace):
    def __init__(self, actions):
        ActionSpace.__init__(self)
        self.actions = actions

    def __len__(self):
        return len(self.actions)

    def index(self, action):
        return self.actions.index(action)

    def __getitem__(self, item):
        return self.actions[item]

    def __iter__(self):
        return self.actions.__iter__()

    def next(self):
        return self.actions.next()

    def to_vector(self, action):
        a_id = self.index(action)
        x = np.zeros(len(self))
        x[a_id] = 1.0
        return x
