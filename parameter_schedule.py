"""
rllib - Reinforcement Learning Library

Classes for implementing parameter schedules.

Goker Erdogan
https://github.com/gokererdogan
"""
import numpy as np


class ParameterSchedule(object):
    """
    ParameterSchedule base class. Value of the parameter is returned by the `get_value` method.
    """
    def __init__(self):
        pass

    def get_value(self, agent):
        """
        Return parameter value for agent. agent is required to access information on learning progress, e.g., training
        epochs elapsed.

        Parameters:
            agent (Agent): Agent being trained.

        Returns:
            float: Parameter value
        """
        raise NotImplementedError()


class GreedyEpsilonConstantSchedule(ParameterSchedule):
    """
    GreedyEpsilonConstantSchedule implements a constant epsilon (probability of picking a random action) schedule.
    """
    def __init__(self, eps):
        """
        Initialize the schedule.

        Parameters:
            eps (float): epsilon value.
        """
        ParameterSchedule.__init__(self)
        if eps < 0.0 or eps > 1.0:
            raise ValueError("Epsilon needs to be between 0.0 and 1.0.")
        self.eps = eps

    def get_value(self, agent):
        return self.eps


class GreedyEpsilonLinearSchedule(ParameterSchedule):
    """
    GreedyEpsilonLinearSchedule implements an epsilon schedule that decreases (or increases) linearly over time.
    """
    def __init__(self, start_eps, end_eps, no_episodes, decrease_period):
        """
        Parameters:
            start_eps (float): Start epsilon value.
            end_eps (float): Final epsilon value.
            no_episodes (int): Number of total training episodes.
            decrease_period (int): Epsilon change period. Epsilon is decreased (or increased) every decrease_period
                episodes.
        """
        ParameterSchedule.__init__(self)
        if start_eps < 0.0 or start_eps > 1.0:
            raise ValueError("Epsilon needs to be between 0.0 and 1.0.")
        if end_eps < 0.0 or end_eps > 1.0:
            raise ValueError("Epsilon needs to be between 0.0 and 1.0.")
        self.start_eps = start_eps
        self.end_eps = end_eps
        self.no_episodes = no_episodes
        self.decrease_period = decrease_period
        self.schedule = np.linspace(self.start_eps, self.end_eps, int(np.ceil(float(no_episodes) / decrease_period)))

    def get_value(self, agent):
        i = int(np.floor(float(agent.episodes_experienced) / self.decrease_period))
        # if we are out of bounds (past the last episode), return the last value
        if i >= len(self.schedule):
            return self.schedule[-1]

        return self.schedule[i]


