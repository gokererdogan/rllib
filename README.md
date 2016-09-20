# rllib - Reinforcement Learning Library

This library implements various reinforcement learning techniques, 
including

- Dynamic programming methods to evaluate policies and find optimal
policies
- A Monte Carlo procedure for evaluating policies
- Q-learning with lookup tables and function approximators (including
neural networks, implemented using theano)
- Policy gradient for multinomial and normal action distributions
with policy functions implemented by neural networks (using theano)

The library also contains an implementation of a game environment 
(GameEnvironment) for using these methods in two-player games, but
these are not tested well at the moment.
