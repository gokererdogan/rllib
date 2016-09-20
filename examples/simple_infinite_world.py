"""
rllib - Reinforcement Learning Library

A simple example of a problem with an infinite action space.

Goker Erdogan
https://github.com/gokererdogan
"""
import numpy as np

from lasagne.updates import sgd

from gmllib.helpers import progress_bar

from rllib.environment import Environment
from rllib.space import RealStateSpace, RealActionSpace
from rllib.parameter_schedule import GreedyEpsilonConstantSchedule
from rllib.policy_gradient import PolicyGradientAgent, PolicyNeuralNetworkNormal


class SimpleInfiniteWorldStateSpace(RealStateSpace):
    def __init__(self):
        RealStateSpace.__init__(self)

    def shape(self):
        return 1,

    def get_initial_state(self):
        return np.array([np.random.randn() * 5.0])

    def get_reward(self, state):
        return -np.sum(np.abs(state))

    def is_goal_state(self, state):
        return False

    def to_vector(self, state):
        return state


class SimpleInfiniteWorldActionSpace(RealActionSpace):
    def __init__(self):
        RealActionSpace.__init__(self)

    def shape(self):
        return 1,

    def to_vector(self, action):
        return action


class SimpleInfiniteWorldEnvironment(Environment):
    def __init__(self):
        Environment.__init__(self, SimpleInfiniteWorldStateSpace())

    def _advance(self, action):
        return self.current_state + action

if __name__ == "__main__":
    env = SimpleInfiniteWorldEnvironment()
    action_space = SimpleInfiniteWorldActionSpace()
    eps = GreedyEpsilonConstantSchedule(eps=0.0)

    policy_function = PolicyNeuralNetworkNormal([], env.state_space, action_space, learning_rate=1e-3, optimizer=sgd,
                                                cov_type='identity')
    pg_learner = PolicyGradientAgent(policy_function, discount_factor=0.9, greed_eps=eps, update_freq=1000)

    epoch_count = 100
    episodes_per_epoch = 5000
    episode_length = 10
    for e in range(epoch_count):
        tot_r = 0.0
        for i in range(episodes_per_epoch):
            progress_bar(i+1, max=episodes_per_epoch, update_freq=episodes_per_epoch/100)
            s, a, r = env.run(pg_learner, episode_length=episode_length)
            tot_r += np.sum(r)
        tot_r /= episodes_per_epoch
        print "Epoch {0:d}, reward per episode: {1:f}".format(e+1, tot_r)

    print "Network parameters:"
    print policy_function.nn.W.get_value()
    print policy_function.nn.b.get_value()

    """
    # test gradients

    episode_length = 1
    reps = 1000

    policy_function = PolicyNeuralNetworkNormal([], env.state_space, action_space, learning_rate=1e-3, optimizer=sgd)
    pg_learner = PolicyGradientAgent(policy_function, discount_factor=0.9, greed_eps=eps, update_freq=reps * 2)

    w = policy_function.nn.W.get_value()
    w[0][0] = 0.5
    w[0][1] = 0.001
    policy_function.nn.W.set_value(w)
    b = policy_function.nn.b.get_value()
    dw = 1e-3

    states = []
    actions = []
    rewards = []
    # use policy gradient to estimate
    expected_tdw = 0.0
    for i in range(reps):
        progress_bar(i+1, max=reps, update_freq=reps/100)
        s, a, r = env.run(pg_learner, episode_length=episode_length)
        states.extend(s[0])
        actions.extend(a[0])
        rewards.extend(r[0:2])
        x = s[0]
        a = a[0]
        r = r[0] + (0.9*r[1])
        m = x*0.5
        logs = x*0.001
        s = np.exp(logs)
        dlogp = x*(a-m)/s**2
        drdw = dlogp*r
        tdw = pg_learner.policy_function.total_grads[0].get_value()
        expected_tdw += drdw
        # print tdw[0][0], expected_tdw

    print policy_function.total_grads[0].get_value()[0][0] / reps, expected_tdw / reps

    # estimate from the same sample
    reward_ss = 0.0
    reward_wpdw_ss = 0.0
    drdw_ss = 0.0
    for i in range(len(states)):
        x = states[i]
        a = actions[i]
        # r = rewards[2*i] + (0.9*rewards[2*i+1])
        r = -np.abs(x) + (0.9*-np.abs(x+a))
        dlogp = 0.0
        totr = 0.0
        for t in range(episode_length):
            m = x*0.5
            mp = x*(0.5+dw)
            logs = x*0.001
            s = np.exp(logs)
            totr += r
            reward_ss += r
            weight = np.exp((2*a - (mp+m))*(mp-m) / (2*s**2))
            reward_wpdw_ss += (r*weight)
            dlogp += x*(a-m)/s**2
            x += a
        drdw_ss += (dlogp*totr)
    reward_ss /= reps
    reward_wpdw_ss /= reps
    drdw_ss /= reps
    drdw_ss2 = (reward_wpdw_ss - reward_ss) / dw
    print drdw_ss
    print drdw_ss2

    # estimate gradient
    # reparameterization trick (noise is fixed, reward changes.)
    reward_w2 = 0.0
    reward_wpdw2 = 0.0
    for i in range(reps):
        x = env.state_space.get_initial_state()
        for _ in range(episode_length):
            m = x*0.5
            m_wpdw = x*(0.5+dw)
            logs = x*0.001
            s = np.exp(logs)
            e = np.random.randn()
            a = m + e*s
            a_wpdw = m_wpdw + e*s
            r = (0.9*-np.abs(x+a))-np.abs(x)
            reward_w2 += r
            r_wpdw = (0.9*-np.abs(x+a_wpdw))-np.abs(x)
            reward_wpdw2 += r_wpdw
            x += a
    reward_w2 /= reps
    reward_wpdw2 /= reps
    drdw2 = (reward_wpdw2 - reward_w2) / dw
    print drdw2

    # estimate gradient 2
    # keep state (reward) constant; probability of state changes.
    reward_w3 = 0.0
    reward_wpdw3 = 0.0
    drdw4 = 0.0
    for i in range(reps):
        x = env.state_space.get_initial_state()
        dlogp = 0.0
        totr = 0.0
        for _ in range(episode_length):
            m = x*0.5
            mp = x*(0.5+dw)
            logs = x*0.001
            s = np.exp(logs)
            e = np.random.randn()
            a = m + e*s
            r = (0.9*-np.abs(x+a))-np.abs(x)
            totr += r
            reward_w3 += r
            weight = np.exp((2*a - (mp+m))*(mp-m) / (2*s**2))
            reward_wpdw3 += (r*weight)
            dlogp += x*(a-m)/s**2
            x += a
        drdw4 += (dlogp*totr)
    reward_w3 /= reps
    reward_wpdw3 /= reps
    drdw4 /= reps
    drdw3 = (reward_wpdw3 - reward_w3) / dw
    print drdw3
    print drdw4
    """

