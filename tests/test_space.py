"""
rllib - Reinforcement Learning Library

Unit tests for space module.

Goker Erdogan
https://github.com/gokererdogan/
"""
import numpy as np
import unittest

from mcmclib.hypothesis import Hypothesis

from rllib.examples.two_state_world import TwoStateSpace, TwoStateInfiniteSpace, TwoStateActionSpace
from rllib.space import RealStateSpace, RealActionSpace, MHStateSpace


class DummyHypothesis(Hypothesis):
    def __init__(self, dummy_var):
        Hypothesis.__init__(self)
        self.dummy_var = dummy_var

    def log_likelihood(self, data=None):
        return 1.0

    def log_prior(self):
        return 2.0

    def render(self):
        return np.ones((5, 5))


class DummyRealStateSpace(RealStateSpace):
    def __init__(self):
        RealStateSpace.__init__(self)

    def is_goal_state(self, state):
        return False

    def get_reward(self, state):
        return np.sum(state)

    def get_initial_state(self):
        return np.random.rand(3)

    def shape(self):
        return 3,


class DummyRealActionSpace(RealActionSpace):
    def __init__(self):
        RealActionSpace.__init__(self)

    def shape(self):
        return 2,


class TestFiniteStateSpace(unittest.TestCase):
    def test_all(self):
        s = TwoStateSpace()
        self.assertEqual(len(s), 2)
        self.assertEqual(s.index(0), 0)
        self.assertEqual(s.index(1), 1)
        self.assertEqual(s[0], 0)
        self.assertEqual(s[1], 1)
        self.assertListEqual([i for i in s], [0, 1])
        self.assertTrue(np.allclose(s.to_vector(0), np.array([1.0, 0.0])))
        self.assertTrue(np.allclose(s.to_vector(1), np.array([0.0, 1.0])))
        self.assertEqual(s.get_reward(0), -0.2)
        self.assertEqual(s.get_reward(1), 1.0)
        self.assertEqual(s.get_initial_state(), 0)
        self.assertTrue(s.is_goal_state(1))
        self.assertFalse(s.is_goal_state(0))
        self.assertEqual(s.to_string(0), '0')
        self.assertEqual(s.to_string(1), '1')

        s = TwoStateInfiniteSpace()
        self.assertEqual(len(s), 2)
        self.assertEqual(s.index(0), 0)
        self.assertEqual(s.index(1), 1)
        self.assertEqual(s[0], 0)
        self.assertEqual(s[1], 1)
        self.assertListEqual([i for i in s], [0, 1])
        self.assertTrue(np.allclose(s.to_vector(0), np.array([1.0, 0.0])))
        self.assertTrue(np.allclose(s.to_vector(1), np.array([0.0, 1.0])))
        self.assertEqual(s.get_reward(0), -0.2)
        self.assertEqual(s.get_reward(1), 1.0)
        self.assertEqual(s.get_initial_state(), 0)
        self.assertFalse(s.is_goal_state(1))
        self.assertFalse(s.is_goal_state(0))
        self.assertEqual(s.to_string(0), '0')
        self.assertEqual(s.to_string(1), '1')


class TestRealStateSpace(unittest.TestCase):
    def test_all(self):
        s = DummyRealStateSpace()
        self.assertEqual(s.shape(), (3,))
        self.assertTrue(np.allclose(s.to_vector(np.array([1.0, 2.0, 3.0])), np.array([1.0, 2.0, 3.0])))
        self.assertEqual(s.get_reward(np.array([2.0, -1.0, 4.0])), 5.0)
        self.assertFalse(s.is_goal_state(np.random.rand(3)))
        self.assertEqual(s.get_initial_state().shape, (3,))


class TestMHStateSpace(unittest.TestCase):
    def test_init(self):
        ss = MHStateSpace(DummyHypothesis, np.zeros((5, 5)), reward_type='log_p')
        self.assertEqual(ss.hypothesis_class, DummyHypothesis)
        self.assertTrue(np.allclose(ss.data, 0.0))
        self.assertEqual(ss.reward_type, 'log_p')
        self.assertEqual(type(ss.hypothesis_params), dict)
        self.assertEqual(len(ss.hypothesis_params), 0)

    def test_get_initial_state(self):
        ss = MHStateSpace(DummyHypothesis, np.zeros((5, 5)), reward_type='log_p', dummy_var='xxx')
        s = ss.get_initial_state()
        self.assertEqual(s['hypothesis'].dummy_var, 'xxx')
        self.assertEqual(s['is_accepted'], True)
        self.assertEqual(s['log_p_increase'], 0.0)
        self.assertEqual(len(s), 3)

    def test_get_reward(self):
        ss = MHStateSpace(DummyHypothesis, np.zeros((5, 5)), reward_type='log_p', dummy_var='xxx')
        h = DummyHypothesis(dummy_var='xxx')
        s = {'hypothesis': h, 'is_accepted': False, 'log_p_increase': 1.5}
        self.assertEqual(ss.get_reward(s), 3.0)
        ss.reward_type = 'acceptance'
        self.assertEqual(ss.get_reward(s), 0)
        ss.reward_type = 'log_p_increase'
        self.assertEqual(ss.get_reward(s), 1.5)
        ss.reward_type = 'abc'
        self.assertRaises(ValueError, ss.get_reward, s)

    def test_is_goal_state(self):
        ss = MHStateSpace(DummyHypothesis, np.zeros((5, 5)), reward_type='log_p', dummy_var='xxx')
        s = {'hypothesis': None, 'is_accepted': True, 'log_p_increase': 0.0}
        self.assertFalse(ss.is_goal_state(s))

    def test_to_vector(self):
        d = np.random.rand(5, 5)
        ss = MHStateSpace(DummyHypothesis, d, reward_type='log_p')
        h = DummyHypothesis(dummy_var='xxx')
        s = {'hypothesis': h, 'is_accepted': False, 'log_p_increase': 1.5}
        self.assertTrue(np.allclose(ss.to_vector(s), 1.0-d))


class TestFiniteActionSpace(unittest.TestCase):
    def test_all(self):
        a = TwoStateActionSpace()
        self.assertEqual(len(a), 2)
        self.assertEqual(a.index('L'), 0)
        self.assertEqual(a.index('R'), 1)
        self.assertEqual(a[0], 'L')
        self.assertEqual(a[1], 'R')
        self.assertListEqual([i for i in a], ['L', 'R'])
        self.assertTrue(np.allclose(a.to_vector('L'), np.array([1.0, 0.0])))
        self.assertTrue(np.allclose(a.to_vector('R'), np.array([0.0, 1.0])))
        self.assertEqual(a.to_string('L'), 'L')
        self.assertEqual(a.to_string('R'), 'R')


class TestRealActionSpace(unittest.TestCase):
    def test_all(self):
        a = DummyRealActionSpace()
        self.assertEqual(a.shape(), (2,))
        self.assertTrue(np.allclose(a.to_vector(np.array([0.32, 0.55])), np.array([0.32, 0.55])))
