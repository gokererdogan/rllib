import numpy as np

from mcmclib.hypothesis import Hypothesis

from rllib.parameter_schedule import GreedyEpsilonConstantSchedule
from rllib.q_learning import QLearningAgent, QTableLookup
from rllib.environment import MHEnvironment
from rllib.space import MHStateSpace, FiniteStateSpace, FiniteActionSpace


class TwoStateHypothesis(Hypothesis):
    def __init__(self, state=0):
        Hypothesis.__init__(self)
        self.state = state

    def log_prior(self):
        return 0.0

    def log_likelihood(self, data=None):
        if self.state == 0:
            return np.log(0.4)
        elif self.state == 1:
            return np.log(0.6)
        elif self.state == -1:
            return -100.0
        raise ValueError("State should be 0 or 1.")


class TwoStateMHStateSpace(MHStateSpace, FiniteStateSpace):
    def __init__(self, reward_type):
        FiniteStateSpace.__init__(self, states=[-1, 0, 1])
        MHStateSpace.__init__(self, TwoStateHypothesis, data=None, reward_type=reward_type)

    def index(self, state):
        return FiniteStateSpace.index(self, state['hypothesis'].state)

    def to_vector(self, state):
        return FiniteStateSpace.to_vector(self, state)


class TwoStateMHActionSpace(FiniteActionSpace):
    def __init__(self):
        FiniteActionSpace.__init__(self, actions=['L', 'R'])

    def reverse(self, action):
        if action == 'L':
            return 'R'
        else:
            return 'L'


class TwoStateMHEnvironment(MHEnvironment):
    def __init__(self, state_space):
        MHEnvironment.__init__(self, state_space)

    def _apply_action_to_hypothesis(self, hypothesis, action):
        if hypothesis.state == 0:
            if action == 'L':
                return TwoStateHypothesis(state=-1)
            elif action == 'R':
                return TwoStateHypothesis(state=1)
        elif hypothesis.state == 1:
            if action == 'L':
                return TwoStateHypothesis(state=0)
            elif action == 'R':
                return TwoStateHypothesis(state=-1)
        elif hypothesis.state == -1:
            raise RuntimeError("This should never happen!")

        raise ValueError("Unknown state or action: {0:s}".format(action))


if __name__ == "__main__":
    ts_state_space = TwoStateMHStateSpace(reward_type='log_p')
    ts_action_space = TwoStateMHActionSpace()
    ts_env = TwoStateMHEnvironment(ts_state_space)

    eps_schedule = GreedyEpsilonConstantSchedule(eps=0.2)
    q_function = QTableLookup(ts_state_space, ts_action_space, learning_rate=0.05)
    # q_function.q_table = np.array([[0.0, 0.0], [0.0, 1.0], [0.0, 1.0]])
    q_agent = QLearningAgent(q_function, discount_factor=0.9, greed_eps=eps_schedule)

    states, actions, rewards = ts_env.run(q_agent, episode_length=100000)
    print q_function.q_table

    print np.sum([s['hypothesis'].state for s in states])
