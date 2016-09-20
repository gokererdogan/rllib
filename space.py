"""
rllib - Reinforcement Learning Library

StateSpace and ActionSpace classes.

Goker Erdogan
https://github.com/gokererdogan
"""

import numpy as np


class StateSpace(object):
    """
    StateSpace base class. This abstract class specifies the interface expected from state space classes.
    """
    def __init__(self):
        pass

    def get_reward(self, state):
        """
        Return the reward associated with state.
        """
        raise NotImplementedError

    def get_initial_state(self):
        """
        Return the initial state.
        """
        raise NotImplementedError

    def is_goal_state(self, state):
        """
        Return True if state is a goal state.
        """
        raise NotImplementedError()

    def to_string(self, state):
        return str(state)

    def to_vector(self, state):
        """Convert the state to a real vector representation. This is required for function approximators like neural
        networks.
        """
        pass


class RealStateSpace(object):
    """
    RealStateSpace base class. This abstract class specifies the interface expected from state spaces over real numbers,
    where a state is represented by a real vector (or any real tensor) of fixed dimensionality.
    """
    def __init__(self):
        pass

    def shape(self):
        """
        Return the shape of the real tensor representing a state.
        """
        raise NotImplementedError()

    def get_reward(self, state):
        """
        Return the reward associated with state.
        """
        raise NotImplementedError

    def get_initial_state(self):
        """
        Return the initial state.
        """
        raise NotImplementedError

    def is_goal_state(self, state):
        """
        Return True if state is a goal state.
        """
        raise NotImplementedError()

    def to_string(self, state):
        return str(state)

    def to_vector(self, state):
        """Convert the state to a real vector representation. This is required for function approximators like neural
        networks.
        """
        return state


class FiniteStateSpace(StateSpace):
    """
    FiniteStateSpace class. This class implements a generic finite state space that can be constructed simply
    from a list of possible states. This class implements the container (list) interface, hence can be used as a list.
    """
    def __init__(self, states):
        """
        Parameters:
            states (list): List of possible states
        """
        StateSpace.__init__(self)
        self.states = states

    def __len__(self):
        """
        Return the number of states in state space.

        Returns:
            int: Number of states.
        """
        return len(self.states)

    def index(self, state):
        """
        Return the index of state. This enables one to use states as indices to an array like data structure.
        """
        return self.states.index(state)

    def __getitem__(self, index):
        """
        Return the state with the given index.

        Parameters:
            index (int): State index

        Returns:
            object: State with the given index.
        """
        return self.states[index]

    def __iter__(self):
        """
        Return iterator over states.
        """
        return self.states.__iter__()

    def next(self):
        """
        Return next state.
        """
        return self.states.next()

    def get_reward(self, state):
        raise NotImplementedError()

    def get_initial_state(self):
        raise NotImplementedError()

    def is_goal_state(self, state):
        raise NotImplementedError()

    def to_vector(self, state):
        """
        Convert the state to a real vector representation. We use 1-of-K encoding to represent states as vectors.

        Parameters:
            state

        Returns:
            numpy.ndarray: 1-of-K encoded vector of state
        """
        s_id = self.index(state)
        x = np.zeros(len(self))
        x[s_id] = 1.0
        return x


class ActionSpace(object):
    """
    ActionSpace base class. This abstract class specifies the interface expected from action space classes.
    """
    def __init__(self):
        pass

    def to_string(self, action):
        return str(action)

    def to_vector(self, action):
        """Convert the action to a real vector representation. This is required for function approximators like neural
        networks.
        """
        pass


class RealActionSpace(ActionSpace):
    """
    RealActionSpace base class. This abstract class specifies the interface expected from an action space over reals,
    i.e., a space where each action is represented by a real vector (or tensor).
    """
    def __init__(self):
        ActionSpace.__init__(self)

    def shape(self):
        """
        Return the shape of the tensor representing an action.
        """
        raise NotImplementedError()

    def to_vector(self, action):
        return action


class FiniteActionSpace(ActionSpace):
    """
    FiniteActionSpace class. This class implements a generic discrete action space that can be constructed simply
    from a list of actions. This class implements the container (list) interface, hence can be used like a list."""
    def __init__(self, actions):
        ActionSpace.__init__(self)
        self.actions = actions

    def __len__(self):
        """
        Returns number of possible actions.

        Returns:
            int: Number of possible actions
        """
        return len(self.actions)

    def index(self, action):
        """
        Returns the index for action. This is useful if an array like structure needs to be accessed using actions.

        Parameters:
            action

        Returns:
            int: Index for action
        """
        return self.actions.index(action)

    def __getitem__(self, index):
        """
        Return the action with the given index.

        Parameters:
            index (int): Action index

        Returns:
            object: Action with the given index.
        """
        return self.actions[index]

    def __iter__(self):
        """
        Return iterator over actions.
        """
        return self.actions.__iter__()

    def next(self):
        """
        Return the next action.
        """
        return self.actions.next()

    def to_vector(self, action):
        """
        Convert the action to a real vector representation. We use 1-of-K encoding to represent actions as vectors.

        Parameters:
           action

        Returns:
            numpy.ndarray: 1-of-K encoded vector of action
        """
        a_id = self.index(action)
        x = np.zeros(len(self))
        x[a_id] = 1.0
        return x
