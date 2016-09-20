"""
rllib - Reinforcement Learning Library

Unit tests for environment module.

Goker Erdogan
https://github.com/gokererdogan/
"""
import unittest
import numpy as np
from rllib.examples.two_state_world import TwoStateFiniteWorldEnvironment, \
    TwoStateInfiniteWorldEnvironment, TwoStateWorldAgent


class TestEnvironment(unittest.TestCase):
    def test_init(self):
        e = TwoStateFiniteWorldEnvironment()
        self.assertIsNotNone(e.state_space)
        self.assertIsNone(e.current_state)
        self.assertIsNone(e.current_reward)

    def test_reset(self):
        e = TwoStateFiniteWorldEnvironment()
        e.reset()
        self.assertEqual(e.current_state, 0)
        self.assertEqual(e.current_reward, -0.2)

    def test_run(self):
        e = TwoStateFiniteWorldEnvironment()
        a = TwoStateWorldAgent()
        states, actions, rewards = e.run(a, episode_length=100)
        self.assertEqual(len(states), len(actions))
        self.assertEqual(len(actions), len(rewards))
        self.assertIsNone(actions[-1])
        # all states except last are 0
        self.assertTrue(np.all(states[0:-1] == 0))
        self.assertTrue(np.all(actions[0:-2] == 'L'))
        self.assertTrue(np.allclose(rewards[0:-1], -0.2))
        # last state is goal state
        self.assertEqual(states[-1], 1)
        self.assertEqual(actions[-2], 'R')
        self.assertEqual(rewards[-1], 1.0)

        e = TwoStateInfiniteWorldEnvironment()
        states, actions, rewards = e.run(a, episode_length=100)
        self.assertEqual(len(states), len(actions))
        self.assertEqual(len(actions), len(rewards))
        self.assertIsNone(actions[-1])
        # all 0 states have reward -0.2, 1 states have reward 1.0
        self.assertTrue(np.all(rewards[states == 0] == -0.2))
        self.assertTrue(np.all(rewards[states == 1] == 1.0))
        # check actions
        state_transition_action = {(0, 0): 'L', (0, 1): 'R', (1, 0): 'L', (1, 1): 'R'}
        expected_actions = [state_transition_action[a] for a in zip(states[0:-1], states[1:])]
        self.assertListEqual(list(actions[0:-1]), expected_actions)

    def test_get_available_actions(self):
        e = TwoStateFiniteWorldEnvironment()
        a = TwoStateWorldAgent()
        self.assertListEqual(list(e.get_available_actions(a)), ['L', 'R'])
