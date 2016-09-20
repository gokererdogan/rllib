"""
rllib - Reinforcement Learning Library

Monte Carlo and dynamic programming methods for policy evaluation and learning optimal policy.

Goker Erdogan
https://github.com/gokererdogan
"""

import numpy as np


def evaluate_policy_monte_carlo(environment, agent, episode_count, episode_length, discount_factor):
    """
    Evaluate policy (implemented by agent) using Monte Carlo technique. Samples episodes and uses these to estimate
    the Q-function. Note that this method is only applicable to problems with discrete state and action spaces.

    Parameters:
        environment (Environment)
        agent (Agent): Agent implementing the policy to evaluate
        episode_count: Number of sample episodes
        episode_length: Maximum length of an episode
        discount_factor: Reward discounting factor

    Returns:
        numpy.ndarray: Q-function table containing the value of each state, action pair.
    """
    q = np.zeros((len(environment.state_space), len(agent.action_space)))
    episode_count_per_state_action = np.zeros_like(q, dtype=int)

    for e in range(episode_count):
        # sample episode
        states, actions, rewards = environment.run(agent=agent, episode_length=episode_length)
        current_episode_length = len(states)

        # calculate counts and total_reward for each state, action pair
        for s_i, state in enumerate(environment.state_space):
            for a_i, action in enumerate(agent.action_space):
                # get the first timestep with state, action pair
                start_ix = _get_timestep_with_state_action(states, actions, state, action)
                if start_ix is not None:
                    episode_count_per_state_action[s_i, a_i] += 1
                    q[s_i, a_i] += np.sum(rewards[start_ix:] * np.power(discount_factor,
                                                                        range(current_episode_length - start_ix)))

    # prevent division by zero
    episode_count_per_state_action[episode_count_per_state_action == 0] = 1
    q /= episode_count_per_state_action

    return q


def _get_timestep_with_state_action(states, actions, state, action):
    """
    Returns the first index (timestep) with the given state, action pair.

    Parameters:
        states (list): List of states
        actions (list): List of actions
        state: State we are looking for
        action: Action we are looking for

    Returns:
        int: First index where the given state, action occurs. None if state, action pair cannot be found.
    """
    for i, (s, a) in enumerate(zip(states, actions)):
        # note that we match if action is None as well. action is None only in the terminal state.
        if s == state and (a == action or a is None):
            return i
    return None


def evaluate_policy_dp(environment, agent, discount_factor, eps=1e-6):
    """
    Evaluate policy (implemented by agent) using dynamic programming.
    Note that this method is only applicable to problems with discrete state and action spaces.

    Parameters:
        environment (Environment)
        agent (Agent): Agent implementing the policy to evaluate
        discount_factor: Reward discounting factor
        eps (float): Convergence threshold

    Returns:
        numpy.ndarray: Q-function table containing the value of each state, action pair.
    """
    state_count = len(environment.state_space)
    action_count = len(agent.action_space)

    # randomly initialize
    q = np.random.rand(state_count, action_count) * 0.1

    converged = False
    while not converged:
        tot_change = 0.0
        for state_id in range(state_count):
            state = environment.state_space[state_id]
            for action_id in range(action_count):
                action = agent.action_space[action_id]
                q_sa = 0.0

                # get reward and next states
                reward, next_states, prob_next_states = environment.get_next_states(state, action)

                q_sa += reward

                # for all possible next states
                for sp, prob_sp in zip(next_states, prob_next_states):
                    sp_id = environment.state_space.index(sp)

                    # get action probabilities
                    p_a = agent.get_action_probabilities(sp)
                    q_sa += np.sum(prob_sp * discount_factor * q[sp_id, :] * p_a)

                tot_change += (q_sa - q[state_id, action_id])**2
                q[state_id, action_id] = q_sa

        if tot_change < eps:
            converged = True

    return q


def calculate_optimal_q_dp(environment, action_space, discount_factor, eps=1e-6):
    """
    Calculate optimal Q-function using dynamic programming.
    Note that this method is only applicable to problems with discrete state and action spaces.

    Parameters:
        environment (Environment)
        action_space (ActionSpace)
        discount_factor: Reward discounting factor
        eps (float): Convergence threshold

    Returns:
        numpy.ndarray: Q-function table containing the value of each state, action pair.
    """
    state_count = len(environment.state_space)
    action_count = len(action_space)

    # randomly initialize
    q = np.random.rand(state_count, action_count) * 0.1

    converged = False
    while not converged:
        tot_change = 0.0
        for state_id in range(state_count):
            state = environment.state_space[state_id]
            for action_id in range(action_count):
                action = action_space[action_id]
                q_sa = 0.0

                # get reward and next states
                reward, next_states, prob_next_states = environment.get_next_states(state, action)

                q_sa += reward

                # pick the action with maximum value for each possible next state
                for sp, prob_sp in zip(next_states, prob_next_states):
                    sp_id = environment.state_space.index(sp)

                    # note the max (instead of sum).
                    q_sa += np.max(prob_sp * discount_factor * q[sp_id, :])

                tot_change += (q_sa - q[state_id, action_id])**2
                q[state_id, action_id] = q_sa

        if tot_change < eps:
            converged = True

    return q

