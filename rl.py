import numpy as np


def evaluate_policy_monte_carlo(environment, agent, episode_count, episode_length, discount_factor):
    q = np.zeros((environment.state_count, agent.action_count))
    episode_count_per_state_action = np.zeros_like(q, dtype=int)

    for e in range(episode_count):
        states, actions, rewards = environment.run(agent=agent, episode_length=episode_length)
        current_episode_length = len(states)

        for s_i, state in enumerate(environment.state_space):
            for a_i, action in enumerate(agent.action_space):
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
    for i, (s, a) in enumerate(zip(states, actions)):
        if s == state and (a == action or a is None):
            return i
    return None


def evaluate_policy_dp(environment, agent, discount_factor, eps=1e-6):
    state_count = environment.state_count
    action_count = agent.action_count

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
    state_count = environment.state_count
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

                    q_sa += np.max(prob_sp * discount_factor * q[sp_id, :])

                tot_change += (q_sa - q[state_id, action_id])**2
                q[state_id, action_id] = q_sa

        if tot_change < eps:
            converged = True

    return q

