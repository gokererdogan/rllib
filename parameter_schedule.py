import numpy as np


class ParameterSchedule(object):
    def __init__(self):
        pass

    def get_value(self, agent):
        raise NotImplementedError()


class GreedyEpsilonConstantSchedule(ParameterSchedule):
    def __init__(self, eps):
        ParameterSchedule.__init__(self)
        self.eps = eps

    def get_value(self, agent):
        return self.eps


class GreedyEpsilonLinearSchedule(ParameterSchedule):
    def __init__(self, start_eps, end_eps, no_episodes, decrease_period):
        ParameterSchedule.__init__(self)
        self.start_eps = start_eps
        self.end_eps = end_eps
        self.no_episodes = no_episodes
        self.decrease_period = decrease_period
        self.schedule = np.linspace(self.start_eps, self.end_eps, int(np.ceil(float(no_episodes) / decrease_period)))

    def get_value(self, agent):
        i = int(np.floor(float(agent.episodes_experienced) / self.decrease_period))
        return self.schedule[i]


