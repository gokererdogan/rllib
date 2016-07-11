import numpy as np

from rllib.examples.two_state_world import TwoStateInfiniteWorldEnvironment, TwoStateFiniteWorldEnvironment
from rllib.policy_gradient import PolicyGradientAgent
from rllib.parameter_schedule import GreedyEpsilonConstantSchedule

env = TwoStateInfiniteWorldEnvironment()
pgl = PolicyGradientAgent(env.state_space, env.action_space, greed_eps=GreedyEpsilonConstantSchedule(0.0),
                          learning_rate=0.0, update_freq=100000, optimizer='gd', apply_baseline=False,
                          clip_gradients=False)

pgl.wa.set_value(np.zeros((2, 2)))
pgl.ba.set_value(np.zeros(2))

wai = pgl.wa.get_value().copy()
bai = pgl.ba.get_value().copy()

el = 5

reps = 20000
avg_r = 0.0
for i in range(reps):
    s, a, r = env.run(pgl, el)
    avg_r += np.sum(r) / (len(a) - 1)
avg_r /= reps
print avg_r

dwa = -pgl.total_dwa.get_value().copy() / reps
dba = -pgl.total_dba.get_value().copy() / reps

print dwa
print dba

"""
# estimate by finite differences
h = 1e-1
pgl.set_learning_mode(False)

dwa_f = np.zeros((2, 2))
dba_f = np.zeros(2)
for i in range(2):
    ban = bai.copy()
    ban[i] += h
    pgl.ba.set_value(ban)
    avg_ri = 0.0
    for _ in range(reps):
        s, a, r = env.run(pgl, el)
        avg_ri += (np.sum(r) / (len(a) - 1))
    avg_ri /= reps
    print avg_ri
    dba_f[i] = (avg_ri - avg_r) / h
    pgl.ba.set_value(bai)
    for j in range(2):
        wan = wai.copy()
        wan[i, j] += h
        pgl.wa.set_value(wan)
        avg_rij = 0.0
        for _ in range(reps):
            s, a, r = env.run(pgl, el)
            avg_rij += (np.sum(r) / (len(a) - 1))
        avg_rij /= reps
        print avg_rij
        dwa_f[i, j] = (avg_rij - avg_r) / h
        pgl.wa.set_value(wai)

print dwa_f
print dba_f
"""