"""
rllib - Reinforcement Learning Library

Unit tests for agent module.

Goker Erdogan
https://github.com/gokererdogan/
"""
import unittest
from rllib.agent import *


class DummyAgent(Agent):
    def __init__(self):
        Agent.__init__(self, action_space=[0, 1])

    def reset(self):
        pass

    def get_action(self, state, available_actions=None):
        return 0

    def get_action_probability(self, state, action=None):
        if action is None:
            return [0.5, 0.5]
        else:
            return 0.5

    def perceive(self, state, reward, available_actions, reached_goal_state=False, episode_end=False):
        return self.get_action(state)


class TestAgent(unittest.TestCase):
    def test_init(self):
        a = DummyAgent()
        self.assertIsNotNone(a.action_space)
        self.assertTrue(a.learning_on)

    def test_get_action(self):
        a = DummyAgent()
        self.assertEqual(a.get_action(None), 0)

    def test_get_action_probability(self):
        a = DummyAgent()
        self.assertEqual(a.get_action_probability(None, 0), 0.5)
        self.assertListEqual(a.get_action_probability(None, None), [0.5, 0.5])

    def test_perceive(self):
        a = DummyAgent()
        self.assertEqual(a.perceive(None, 0.0, []), 0)

    def test_set_learning_mode(self):
        a = DummyAgent()
        a.set_learning_mode(False)
        self.assertFalse(a.learning_on)
        a.set_learning_mode(True)
        self.assertTrue(a.learning_on)
