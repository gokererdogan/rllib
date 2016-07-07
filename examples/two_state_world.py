import numpy as np

from gmllib.helpers import progress_bar

from rllib.environment import Environment
from rllib.agent import Agent
from rllib.space import DiscreteStateSpace, DiscreteActionSpace
from rllib.parameter_schedule import GreedyEpsilonConstantSchedule
from rllib.q_learning import QLearningAgent
from rllib.rl import evaluate_policy_dp, evaluate_policy_monte_carlo, calculate_optimal_q_dp
from rllib.policy_gradient import PolicyGradientAgent


class TwoStateSpace(DiscreteStateSpace):
    def __init__(self):
        DiscreteStateSpace.__init__(self, states=[0, 1])

        self.rewards = [-0.2, 1.0]

    def get_reward(self, state):
        return self.rewards[state]

    def get_initial_state(self):
        return 0

    def is_goal_state(self, state):
        return state == 1


class TwoStateInfiniteSpace(TwoStateSpace):
    def __init__(self):
        TwoStateSpace.__init__(self)

    def is_goal_state(self, state):
        return False


class TwoStateActionSpace(DiscreteActionSpace):
    def __init__(self):
        DiscreteActionSpace.__init__(self, actions=['L', 'R'])


class TwoStateFiniteWorldEnvironment(Environment):
    def __init__(self):
        Environment.__init__(self, state_space=TwoStateSpace(), action_space=TwoStateActionSpace())

    def _advance(self, action):
        if self.current_state == 0 and action == 'L':
            return 0
        elif self.current_state == 0 and action == 'R':
            return 1
        elif self.current_state == 1:  # terminal state
            return self.current_state

    def get_next_states(self, state, action):
        if state == 0 and action == 'L':
            return self.state_space.get_reward(state), [0], np.array([1.0])
        elif state == 0 and action == 'R':
            return self.state_space.get_reward(state), [1], np.array([1.0])
        elif state == 1:
            return self.state_space.get_reward(state), np.array([]), np.array([])


class TwoStateWorldAgent(Agent):
    def __init__(self):
        Agent.__init__(self, action_space=TwoStateActionSpace())

    def perceive(self, state, reward, available_actions, reached_goal_state=False, episode_end=False):
        return np.random.choice(self.action_space)

    def get_action_probabilities(self, state):
        return np.array([0.5, 0.5])


class TwoStateInfiniteWorldEnvironment(Environment):
    def __init__(self):
        Environment.__init__(self, state_space=TwoStateInfiniteSpace(),
                             action_space=TwoStateActionSpace())

    def _advance(self, action):
        if self.current_state == 0 and action == 'L':
            return 0
        elif self.current_state == 0 and action == 'R':
            return 1
        elif self.current_state == 1 and action == 'L':
            return 0
        elif self.current_state == 1 and action == 'R':
            return 1

    def get_next_states(self, state, action):
        if state == 0 and action == 'L':
            return self.state_space.get_reward(state), [0], np.array([1.0])
        elif state == 0 and action == 'R':
            return self.state_space.get_reward(state), [1], np.array([1.0])
        elif state == 1 and action == 'L':
            return self.state_space.get_reward(state), [0], np.array([1.0])
        elif state == 1 and action == 'R':
            return self.state_space.get_reward(state), [1], np.array([1.0])


if __name__ == "__main__":
    eps_schedule = GreedyEpsilonConstantSchedule(eps=0.2)

    """
    # finite case
    env = TwoStateFiniteWorldEnvironment()
    agent = TwoStateWorldAgent()

    pg_learner = PolicyGradientAgent(env.state_space, agent.action_space, learning_rate=0.1, greed_eps=eps_schedule,
                                     update_freq=1, optimizer='gd')

    for i in range(10000):
        progress_bar(i+1, max=10000, update_freq=100)
        env.run(pg_learner, episode_length=np.inf)

    print pg_learner.forward([1., 0.])
    print pg_learner.forward([0., 1.])
    print

    # calculate q for a given policy
    # expected q: [[0.4, 0.8], [1.0, 1.0]]
    q_mc = evaluate_policy_monte_carlo(env, agent, episode_count=10000, episode_length=np.inf, discount_factor=1.0)
    print q_mc
    print

    q_dp = evaluate_policy_dp(env, agent, discount_factor=1.0, eps=1e-9)
    print q_dp
    print

    # calculate the optimal value function
    # expected q: [[0.6, 0.8], [1.0, 1.0]]
    q_opt_dp = calculate_optimal_q_dp(env, discount_factor=1.0, eps=1e-9)
    print q_opt_dp
    print

    # q-learning
    q_learner = QLearningAgent(env.state_space, agent.action_space, discount_factor=1.0,
                               greed_eps=eps_schedule, learning_rate=0.05)

    for i in range(2000):
        env.run(q_learner, episode_length=200)
    print q_learner.q
    print
    """

    # infinite case (discount factor has to be < 1.0)
    env = TwoStateInfiniteWorldEnvironment()
    agent = TwoStateWorldAgent()

    """
    # calculate q for a given policy
    # expected q: [[2.86, 3.94], [4.06, 5.14]]
    q_mc = evaluate_policy_monte_carlo(env, agent, episode_count=5000, episode_length=200, discount_factor=0.9)
    print q_mc
    print

    q_dp = evaluate_policy_dp(env, agent, discount_factor=0.9, eps=1e-9)
    print q_dp
    print

    # calculate the optimal value function
    # expected q: [[7.72, 8.8], [8.92, 10.0]]
    q_opt_dp = calculate_optimal_q_dp(env, discount_factor=0.9, eps=1e-9)
    print q_opt_dp
    print

    # q-learning
    q_learner = QLearningAgent(env.state_space, agent.action_space, discount_factor=0.9,
                               greed_eps=eps_schedule, learning_rate=0.1)

    for i in range(2000):
        env.run(q_learner, episode_length=100)
    print q_learner.q
    print
    """

    # policy gradient
    # note that applying a reward baseline is important for learning in this case
    pg_learner = PolicyGradientAgent(env.state_space, agent.action_space, learning_rate=0.1, greed_eps=eps_schedule,
                                     update_freq=1, optimizer='gd', apply_baseline=True)

    for e in range(5):
        for i in range(500):
            env.run(pg_learner, episode_length=100)

        print pg_learner.forward([1., 0.])
        print pg_learner.forward([0., 1.])
        print pg_learner.grad_magnitudes[0][-1]
        print pg_learner.grad_magnitudes[1][-1]
        print
