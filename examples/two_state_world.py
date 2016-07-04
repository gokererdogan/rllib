import numpy as np

from gmllib.helpers import progress_bar

from rllib.environment import Environment
from rllib.agent import Agent
from rllib.q_learning import QLearningAgent
from rllib.rl import evaluate_policy_dp, evaluate_policy_monte_carlo, calculate_optimal_q_dp
from rllib.policy_gradient import PolicyGradientAgent


class TwoStateFiniteWorldEnvironment(Environment):
    def __init__(self, punishment=0.1, reward=1.0):
        Environment.__init__(self, state_space=[0, 1], action_space=['L', 'R'], initial_state=0,
                             initial_reward=-punishment, goal_states=[1])

        self.punishment = -punishment
        self.reward = reward

    def _advance(self, action):
        if self.current_state == 0 and action == 'L':
            return 0, self.punishment
        elif self.current_state == 0 and action == 'R':
            return 1, self.reward
        elif self.current_state == 1:  # terminal state
            return self.current_state, 0.0

    def get_next_states(self, state, action):
        if state == 0 and action == 'L':
            return self.punishment, [0], np.array([1.0])
        elif state == 0 and action == 'R':
            return self.punishment, [1], np.array([1.0])
        elif state == 1:
            return self.reward, np.array([]), np.array([])


class TwoStateWorldAgent(Agent):
    def __init__(self):
        Agent.__init__(self, action_space=['L', 'R'])

    def perceive(self, state, reward, reached_goal_state=False):
        return np.random.choice(self.action_space)

    def get_action_probabilities(self, state):
        return np.array([0.5, 0.5])


class TwoStateInfiniteWorldEnvironment(Environment):
    def __init__(self, punishment=0.1, reward=1.0):
        Environment.__init__(self, state_space=[0, 1], action_space=['L', 'R'],
                             initial_state=0, initial_reward=-punishment)

        self.punishment = -punishment
        self.reward = reward

    def _advance(self, action):
        if self.current_state == 0 and action == 'L':
            return 0, self.punishment
        elif self.current_state == 0 and action == 'R':
            return 1, self.reward
        elif self.current_state == 1 and action == 'L':
            return 0, self.punishment
        elif self.current_state == 1 and action == 'R':
            return 1, self.reward

    def get_next_states(self, state, action):
        if state == 0 and action == 'L':
            return self.punishment, [0], np.array([1.0])
        elif state == 0 and action == 'R':
            return self.punishment, [1], np.array([1.0])
        elif state == 1 and action == 'L':
            return self.reward, [0], np.array([1.0])
        elif state == 1 and action == 'R':
            return self.reward, [1], np.array([1.0])


if __name__ == "__main__":
    # finite case
    env = TwoStateFiniteWorldEnvironment(punishment=0.2, reward=1.0)
    agent = TwoStateWorldAgent()

    pg_learner = PolicyGradientAgent(env.state_space, agent.action_space, learning_rate=0.1,
                                     update_freq=10, optimizer='gd')

    for i in range(10000):
        progress_bar(i, max=10000, update_freq=10)
        env.run(pg_learner, episode_length=np.inf)
    """

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
    q_opt_dp = calculate_optimal_q_dp(env, np.array(['L', 'R']), discount_factor=1.0, eps=1e-9)
    print q_opt_dp
    print

    # q-learning
    q_learner = QLearningAgent(env.state_space, agent.action_space, discount_factor=1.0,
                               greed_eps=0.1, learning_params={'LEARNING_RATE': 0.05})

    for i in range(2000):
        env.run(q_learner, episode_length=200)
    print q_learner.q
    print

    # infinite case (discount factor has to be < 1.0)
    env = TwoStateInfiniteWorldEnvironment(punishment=0.2, reward=1.0)
    agent = TwoStateWorldAgent()

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
    q_opt_dp = calculate_optimal_q_dp(env, np.array(['L', 'R']), discount_factor=0.9, eps=1e-9)
    print q_opt_dp
    print

    # q-learning
    q_learner = QLearningAgent(env.state_space, agent.action_space, discount_factor=0.9,
                               greed_eps=0.1, learning_params={'LEARNING_RATE': 0.1})

    for i in range(2000):
        env.run(q_learner, episode_length=100)
    print q_learner.q
    print

    # policy gradient
    pg_learner = PolicyGradientAgent(env.state_space, agent.action_space, learning_rate=0.01,
                                     update_freq=10, optimizer='gd')

    for i in range(5000):
        progress_bar(i, max=5000, update_freq=10)
        env.run(pg_learner, episode_length=100)
    """

