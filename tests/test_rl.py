"""
rllib - Reinforcement Learning Library

Unit tests for rl module.

Goker Erdogan
https://github.com/gokererdogan/
"""
import sys
import numpy as np
import unittest

from rllib.examples.two_state_world import TwoStateFiniteWorldEnvironment, TwoStateInfiniteWorldEnvironment, \
    TwoStateWorldAgent
from rllib.rl import evaluate_policy_monte_carlo, evaluate_policy_dp, calculate_optimal_q_dp

TOL = 1e-1  # equality tolerance


class TestRL(unittest.TestCase):
    @unittest.skipIf('--skipslow' in sys.argv, "Slow tests are turned off.")
    def test_evaluate_policy_monte_carlo(self):
        a = TwoStateWorldAgent()
        e = TwoStateFiniteWorldEnvironment()
        q = evaluate_policy_monte_carlo(e, a, episode_count=50000, episode_length=np.inf, discount_factor=1.0)
        # expected q: [[0.4, 0.8], [1.0, 1.0]]
        expected_q = np.array([[0.4, 0.8], [1.0, 1.0]])
        self.assertLess(np.sum(np.abs(q - expected_q)), TOL, msg="{0:s}\n{1:s}".format(q, expected_q))

        q = evaluate_policy_monte_carlo(e, a, episode_count=50000, episode_length=np.inf, discount_factor=0.9)
        # expected q: [[0.20909, 0.7], [1.0, 1.0]]
        expected_q = np.array([[0.20909, 0.7], [1.0, 1.0]])
        self.assertLess(np.sum(np.abs(q - expected_q)), TOL, msg="{0:s}\n{1:s}".format(q, expected_q))

        e = TwoStateInfiniteWorldEnvironment()
        q = evaluate_policy_monte_carlo(e, a, episode_count=5000, episode_length=200, discount_factor=0.9)
        # expected q: [[2.86, 3.94], [4.06, 5.14]]
        expected_q = np.array([[2.86, 3.94], [4.06, 5.14]])
        self.assertLess(np.sum(np.abs(q - expected_q)), TOL, msg="{0:s}\n{1:s}".format(q, expected_q))

    def test_evaluate_policy_dp(self):
        a = TwoStateWorldAgent()
        e = TwoStateFiniteWorldEnvironment()
        q = evaluate_policy_dp(e, a, discount_factor=1.0, eps=1e-6)
        # expected q: [[0.4, 0.8], [1.0, 1.0]]
        expected_q = np.array([[0.4, 0.8], [1.0, 1.0]])
        self.assertLess(np.sum(np.abs(q - expected_q)), TOL, msg="{0:s}\n{1:s}".format(q, expected_q))

        q = evaluate_policy_dp(e, a, discount_factor=0.9, eps=1e-6)
        # expected q: [[0.20909, 0.7], [1.0, 1.0]]
        expected_q = np.array([[0.20909, 0.7], [1.0, 1.0]])
        self.assertLess(np.sum(np.abs(q - expected_q)), TOL, msg="{0:s}\n{1:s}".format(q, expected_q))

        e = TwoStateInfiniteWorldEnvironment()
        q = evaluate_policy_dp(e, a, discount_factor=0.9)
        # expected q: [[2.86, 3.94], [4.06, 5.14]]
        expected_q = np.array([[2.86, 3.94], [4.06, 5.14]])
        self.assertLess(np.sum(np.abs(q - expected_q)), TOL, msg="{0:s}\n{1:s}".format(q, expected_q))

    def test_calculate_optimal_q_dp(self):
        a = TwoStateWorldAgent()
        e = TwoStateFiniteWorldEnvironment()
        q = calculate_optimal_q_dp(e, a.action_space, discount_factor=1.0, eps=1e-6)
        # expected q: [[0.6, 0.8], [1.0, 1.0]]
        expected_q = np.array([[0.6, 0.8], [1.0, 1.0]])
        self.assertLess(np.sum(np.abs(q - expected_q)), TOL, msg="{0:s}\n{1:s}".format(q, expected_q))

        q = calculate_optimal_q_dp(e, a.action_space, discount_factor=0.9, eps=1e-6)
        # expected q: [[0.43, 0.7], [1.0, 1.0]]
        expected_q = np.array([[0.43, 0.7], [1.0, 1.0]])
        self.assertLess(np.sum(np.abs(q - expected_q)), TOL, msg="{0:s}\n{1:s}".format(q, expected_q))

        e = TwoStateInfiniteWorldEnvironment()
        q = calculate_optimal_q_dp(e, a.action_space, discount_factor=0.9, eps=1e-6)
        # expected q: [[7.72, 8.8], [8.92, 10.0]]
        expected_q = np.array([[7.72, 8.8], [8.92, 10.0]])
        self.assertLess(np.sum(np.abs(q - expected_q)), TOL, msg="{0:s}\n{1:s}".format(q, expected_q))
