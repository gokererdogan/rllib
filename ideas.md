- implement a global state loop (acting, perceiving etc.) this might be useful
for making sure that we do not run act or perceive methods of an agent multiple
times in a row.
- separate out the policy (actor) from the agent. pass policy to dp and monte carlo methods
- separate out the perceiver (learning routines) from the agent. that would enable plugging in 
any learner to an agent.
- DONE rewards should be tied to state space. we should be able to get the reward associated with a
state directly from state itself
    - this way, we don't need to pass initial reward to Environment. right
    now the initial state can be chosen randomly, but not the initial reward
    (look at reset method)
- Partially DONE. have DiscreteSpace and ContinousSpace classes. state_space, action_space
should be instances of these.
- implement temperature for softmax action selection in policy_gradient
- filtering unavailable actions should be done by ActionSpace, not by 
Agent classes.
- picking a random action should be done by ActionSpace, not by Agent
classes
- who should reset episodes_experienced in Agent? and when should it be
reset? We need to make sure it is reset before we retrain if a parameter
(like epsilon of greedy) depends on episoded experienced.