"""
rllib - Reinforcement Learning Library

Unit tests for parameter_schedule module.

Goker Erdogan
https://github.com/gokererdogan/
"""
import unittest
from rllib.parameter_schedule import *


class DummyAgent(object):
    pass


class TestGreedyEpsilonConstantSchedule(unittest.TestCase):
    def test_init(self):
        self.assertRaises(ValueError, GreedyEpsilonConstantSchedule, -0.2)
        self.assertRaises(ValueError, GreedyEpsilonConstantSchedule, 1.2)
        s = GreedyEpsilonConstantSchedule(0.0)
        self.assertEqual(s.eps, 0.0)
        s = GreedyEpsilonConstantSchedule(1.0)
        self.assertEqual(s.eps, 1.0)
        s = GreedyEpsilonConstantSchedule(0.5)
        self.assertEqual(s.eps, 0.5)

    def test_get_value(self):
        s = GreedyEpsilonConstantSchedule(0.0)
        self.assertEqual(s.get_value(None), 0.0)
        s = GreedyEpsilonConstantSchedule(1.0)
        self.assertEqual(s.get_value(None), 1.0)
        s = GreedyEpsilonConstantSchedule(0.5)
        self.assertEqual(s.get_value(None), 0.5)


class TestGreedyEpsilonLinearSchedule(unittest.TestCase):
    def test_init(self):
        self.assertRaises(ValueError, GreedyEpsilonLinearSchedule, start_eps=-0.2, end_eps=0.5, no_episodes=1000,
                          decrease_period=100)
        self.assertRaises(ValueError, GreedyEpsilonLinearSchedule, start_eps=1.2, end_eps=0.5, no_episodes=1000,
                          decrease_period=100)
        self.assertRaises(ValueError, GreedyEpsilonLinearSchedule, start_eps=0.5, end_eps=-0.8, no_episodes=1000,
                          decrease_period=100)
        self.assertRaises(ValueError, GreedyEpsilonLinearSchedule, start_eps=0.5, end_eps=1.5, no_episodes=1000,
                          decrease_period=100)
        self.assertRaises(ValueError, GreedyEpsilonLinearSchedule, start_eps=-0.5, end_eps=1.5, no_episodes=1000,
                          decrease_period=100)

        s = GreedyEpsilonLinearSchedule(1.0, 0.0, 1000, 100)
        self.assertEqual(s.start_eps, 1.0)
        self.assertEqual(s.end_eps, 0.0)
        self.assertEqual(s.no_episodes, 1000)
        self.assertEqual(s.decrease_period, 100)

    def test_get_value(self):
        s = GreedyEpsilonLinearSchedule(1.0, 0.0, 1100, 100)
        a = DummyAgent()
        a.episodes_experienced = 0
        self.assertEqual(s.get_value(a), 1.0)
        a.episodes_experienced = 99
        self.assertEqual(s.get_value(a), 1.0)
        a.episodes_experienced = 100
        self.assertEqual(s.get_value(a), 0.9)
        a.episodes_experienced = 550
        self.assertEqual(s.get_value(a), 0.5)
        a.episodes_experienced = 1099
        self.assertEqual(s.get_value(a), 0.0)
        a.episodes_experienced = 3000
        self.assertEqual(s.get_value(a), 0.0)

