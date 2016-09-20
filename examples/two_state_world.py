import numpy as np

from lasagne.updates import sgd

from gmllib.helpers import progress_bar

from rllib.environment import Environment
from rllib.agent import Agent
from rllib.space import FiniteActionSpace, FiniteStateSpace
from rllib.parameter_schedule import GreedyEpsilonConstantSchedule, GreedyEpsilonLinearSchedule
from rllib.q_learning import QLearningAgent, QTableLookup, QNeuralNetwork
from rllib.rl import evaluate_policy_dp, evaluate_policy_monte_carlo, calculate_optimal_q_dp
from rllib.policy_gradient import PolicyGradientAgent, PolicyNeuralNetworkMultinomial


class TwoStateSpace(FiniteStateSpace):
    def __init__(self):
        FiniteStateSpace.__init__(self, states=[0, 1])

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


class TwoStateActionSpace(FiniteActionSpace):
    def __init__(self):
        FiniteActionSpace.__init__(self, actions=['L', 'R'])


class TwoStateFiniteWorldEnvironment(Environment):
    def __init__(self):
        Environment.__init__(self, state_space=TwoStateSpace())

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

    def reset(self):
        pass

    def perceive(self, state, reward, available_actions, reached_goal_state=False, episode_end=False):
        return np.random.choice(self.action_space)

    def get_action_probabilities(self, state):
        return np.array([0.5, 0.5])


class TwoStateInfiniteWorldEnvironment(Environment):
    def __init__(self):
        Environment.__init__(self, state_space=TwoStateInfiniteSpace())

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
    action_space = TwoStateActionSpace()

    # finite case
    env = TwoStateFiniteWorldEnvironment()
    agent = TwoStateWorldAgent()

    policy_function = PolicyNeuralNetworkMultinomial([], env.state_space, action_space, learning_rate=0.01,
                                                     optimizer=sgd)
    pg_learner = PolicyGradientAgent(policy_function, discount_factor=1.0, greed_eps=eps_schedule, update_freq=50)

    for i in range(10000):
        progress_bar(i+1, max=10000, update_freq=100)
        env.run(pg_learner, episode_length=np.inf)

    print policy_function._forward(np.array([[1., 0.]]))
    print policy_function._forward(np.array([[0., 1.]]))
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
    q_opt_dp = calculate_optimal_q_dp(env, action_space, discount_factor=1.0, eps=1e-9)
    print q_opt_dp
    print

    # q-learning
    eps_schedule = GreedyEpsilonLinearSchedule(start_eps=1.0, end_eps=0.1, no_episodes=5000, decrease_period=500)
    # q_function = QTableLookup(env.state_space, agent.action_space, learning_rate=0.1)
    q_function = QNeuralNetwork([], env.state_space, agent.action_space, learning_rate=0.01)
    q_learner = QLearningAgent(q_function, discount_factor=1.0, greed_eps=eps_schedule)

    for i in range(10000):
        env.run(q_learner, episode_length=200)

    print q_function.get_q(env.state_space[0])
    print q_function.get_q(env.state_space[1])
    print

    # infinite case (discount factor has to be < 1.0)
    env = TwoStateInfiniteWorldEnvironment()

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
    q_opt_dp = calculate_optimal_q_dp(env, action_space, discount_factor=0.9, eps=1e-9)
    print q_opt_dp
    print

    # q-learning
    q_function = QNeuralNetwork([], env.state_space, agent.action_space, learning_rate=0.01)
    # q_function = QTableLookup(env.state_space, agent.action_space, learning_rate=0.1)
    q_learner = QLearningAgent(q_function, discount_factor=0.9, greed_eps=eps_schedule)

    for i in range(2000):
        env.run(q_learner, episode_length=100)
    print q_function.get_q(env.state_space[0])
    print q_function.get_q(env.state_space[1])
    print

    # policy gradient
    policy_function = PolicyNeuralNetworkMultinomial([], env.state_space, action_space, learning_rate=0.001,
                                                     optimizer=sgd)
    pg_learner = PolicyGradientAgent(policy_function, discount_factor=0.9, greed_eps=eps_schedule, update_freq=50)

    for i in range(1000):
        progress_bar(i+1, max=1000, update_freq=10)
        # we need to keep the episodes short; because as the episodes get longer, the performance difference between
        # policies become smaller (hence we don't converge to any solution, or converge to one random policy)
        env.run(pg_learner, episode_length=10)

    print policy_function._forward(np.array([[1., 0.]]))
    print policy_function._forward(np.array([[0., 1.]]))
    print
