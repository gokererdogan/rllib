"""
rllib - Reinforcement Learning Library

Unit tests for q_learning module.

Goker Erdogan
https://github.com/gokererdogan/
"""
import sys
import numpy as np
import unittest

from rllib.examples.two_state_world import TwoStateFiniteWorldEnvironment, TwoStateInfiniteWorldEnvironment, \
    TwoStateActionSpace
from rllib.parameter_schedule import GreedyEpsilonConstantSchedule
from rllib.q_learning import QTableLookup, QNeuralNetwork, QLearningAgent


TOL = 1e-1  # equality tolerance


class TestQTableLookup(unittest.TestCase):
    def setUp(self):
        self.e = TwoStateFiniteWorldEnvironment()
        self.a = TwoStateActionSpace()
        self.qf = QTableLookup(self.e.state_space, self.a, learning_rate=0.1)

    def test_init(self):
        self.assertEqual(self.qf.state_count, 2)
        self.assertEqual(self.qf.action_count, 2)
        self.assertIsNotNone(self.qf.state_space)
        self.assertIsNotNone(self.qf.action_space)
        self.assertEqual(self.qf.learning_rate, 0.1)
        self.assertEqual(self.qf.q_table.shape, (2, 2))

    def test_get_q(self):
        self.qf.q_table = np.array([[0.1, 0.2], [0.3, 0.4]])
        self.assertEqual(self.qf.get_q(0, 'L'), 0.1)
        self.assertEqual(self.qf.get_q(0, 'R'), 0.2)
        self.assertEqual(self.qf.get_q(1, 'L'), 0.3)
        self.assertEqual(self.qf.get_q(1, 'R'), 0.4)
        self.assertTrue(np.allclose(self.qf.get_q(0), [0.1, 0.2]))
        self.assertTrue(np.allclose(self.qf.get_q(1), [0.3, 0.4]))

    def test_update_q(self):
        q = np.array([[0.1, 0.2], [0.3, 0.4]])
        self.qf.q_table = q
        self.qf.update_q(0.0, 0, 'L')
        self.assertEqual(self.qf.get_q(0, 'L'), 0.09)
        self.qf.update_q(0.0, 0, 'R')
        self.assertEqual(self.qf.get_q(0, 'R'), 0.18)
        self.qf.update_q(1.0, 1, 'L')
        self.assertEqual(self.qf.get_q(1, 'L'), 0.37)
        self.qf.update_q(1.0, 1, 'R')
        self.assertEqual(self.qf.get_q(1, 'R'), 0.46)
        self.qf.update_q(1.0, 0)
        self.assertTrue(np.allclose(self.qf.get_q(0), [0.181, 0.262]))
        self.qf.update_q(0.0, 1)
        self.assertTrue(np.allclose(self.qf.get_q(1), [0.333, 0.414]))


class TestQNeuralNetwork(unittest.TestCase):
    def setUp(self):
        self.e = TwoStateFiniteWorldEnvironment()
        self.a = TwoStateActionSpace()
        # create a single layer neural network
        self.qf = QNeuralNetwork([], self.e.state_space, self.a, learning_rate=0.05)

    def test_init(self):
        self.assertRaises(ValueError, QNeuralNetwork, 'xxx', self.e.state_space, self.a, 0.1)
        self.assertEqual(self.qf.action_count, 2)
        self.assertIsNotNone(self.qf.state_space)
        self.assertIsNotNone(self.qf.action_space)
        self.assertEqual(self.qf.learning_rate, 0.05)
        self.assertEqual(self.qf.nn.W.get_value().shape, (2, 2))
        self.assertEqual(self.qf.nn.b.get_value().shape, (2,))

    def test_get_q(self):
        # w is input_dim x output_dim (not output x input)
        # lasagne DenseLayer calculates x^T*W + b
        self.qf.nn.W.set_value(np.array([[0.0, 0.3], [0.2, 0.5]]))
        self.qf.nn.b.set_value(np.array([0.1, -0.1]))
        self.assertAlmostEqual(self.qf.get_q(0, 'L'), 0.1)
        self.assertAlmostEqual(self.qf.get_q(0, 'R'), 0.2)
        self.assertAlmostEqual(self.qf.get_q(1, 'L'), 0.3)
        self.assertAlmostEqual(self.qf.get_q(1, 'R'), 0.4)
        self.assertTrue(np.allclose(self.qf.get_q(0), [0.1, 0.2]))
        self.assertTrue(np.allclose(self.qf.get_q(1), [0.3, 0.4]))

    def test_update_q(self):
        # w is input_dim x output_dim (not output x input)
        # lasagne DenseLayer calculates x^T*W + b
        self.qf.nn.W.set_value(np.array([[0.0, 0.3], [0.2, 0.5]]))
        self.qf.nn.b.set_value(np.array([0.1, -0.1]))
        self.qf.update_q(0.0, 0, 'L')
        self.assertAlmostEqual(self.qf.get_q(0, 'L'), 0.09)
        self.assertTrue(np.allclose(self.qf.nn.W.get_value(), np.array([[-0.005, 0.3], [0.2, 0.5]])))
        self.assertTrue(np.allclose(self.qf.nn.b.get_value(), np.array([0.095, -0.1])))
        self.qf.update_q(0.0, 0, 'R')
        self.assertAlmostEqual(self.qf.get_q(0, 'R'), 0.18)
        self.assertTrue(np.allclose(self.qf.nn.W.get_value(), np.array([[-0.005, 0.29], [0.2, 0.5]])))
        self.assertTrue(np.allclose(self.qf.nn.b.get_value(), np.array([0.095, -0.11])))
        self.qf.update_q(1.0, 1, 'L')
        self.assertAlmostEqual(self.qf.get_q(1, 'L'), 0.3655)
        self.assertAlmostEqual(self.qf.get_q(1, 'R'), 0.39)
        self.assertTrue(np.allclose(self.qf.nn.W.get_value(), np.array([[-0.005, 0.29], [0.23525, 0.5]])))
        self.assertTrue(np.allclose(self.qf.nn.b.get_value(), np.array([0.13025, -0.11])))
        self.qf.update_q(1.0, 1, 'R')
        self.assertAlmostEqual(self.qf.get_q(1, 'L'), 0.3655)
        self.assertAlmostEqual(self.qf.get_q(1, 'R'), 0.451)
        self.assertTrue(np.allclose(self.qf.nn.W.get_value(), np.array([[-0.005, 0.29], [0.23525, 0.5305]])))
        self.assertTrue(np.allclose(self.qf.nn.b.get_value(), np.array([0.13025, -0.0795])))
        self.qf.update_q(1.0, 0)
        self.assertTrue(np.allclose(self.qf.get_q(0), [0.212725, 0.28945]))
        self.assertTrue(np.allclose(self.qf.get_q(1), [0.4092375, 0.490475]))
        self.assertTrue(np.allclose(self.qf.nn.W.get_value(), np.array([[0.0387375, 0.329475], [0.23525, 0.5305]])))
        self.assertTrue(np.allclose(self.qf.nn.b.get_value(), np.array([0.1739875, -0.040025])))
        self.qf.update_q(0.0, 1)
        self.assertTrue(np.allclose(self.qf.get_q(0), [0.19226313, 0.26492625]))
        self.assertTrue(np.allclose(self.qf.get_q(1), [0.36831375, 0.4414275]))
        self.assertTrue(np.allclose(self.qf.nn.W.get_value(), np.array([[0.0387375, 0.329475], [0.21478812, 0.50597625]])))
        self.assertTrue(np.allclose(self.qf.nn.b.get_value(), np.array([0.15352562, -0.06454875])))


class TestQLearningAgent(unittest.TestCase):
    def setUp(self):
        self.e_fin = TwoStateFiniteWorldEnvironment()
        self.e_inf = TwoStateInfiniteWorldEnvironment()
        self.a = TwoStateActionSpace()
        self.eps = GreedyEpsilonConstantSchedule(0.2)

    def test_init(self):
        qf = QTableLookup(self.e_fin.state_space, self.a, learning_rate=0.01)
        self.assertRaises(ValueError, QLearningAgent, qf, -0.1, self.eps)
        self.assertRaises(ValueError, QLearningAgent, qf, 1.5, self.eps)
        q_learner = QLearningAgent(qf, discount_factor=1.0, greed_eps=self.eps)
        self.assertEqual(q_learner.discount_factor, 1.0)
        self.assertEqual(q_learner.episodes_experienced, 0)
        self.assertIsNone(q_learner.last_state)
        self.assertIsNone(q_learner.last_action)
        self.assertEqual(q_learner.last_reward, 0.0)

    def test_perceive(self):
        qf = QTableLookup(self.e_fin.state_space, self.a, learning_rate=0.1)
        qf.q_table = np.zeros((2, 2))
        q_learner = QLearningAgent(qf, discount_factor=0.9, greed_eps=self.eps)
        q_learner.set_learning_mode(learning_on=False)
        action = q_learner.perceive(0, 0.0, ['L'])

        # test last_state update
        self.assertEqual(q_learner.last_state, 0)
        self.assertEqual(q_learner.last_action, 'L')
        self.assertEqual(q_learner.last_reward, 0.0)

        # test learning mode
        self.assertTrue(np.allclose(q_learner.q.get_q(0), 0.0))
        self.assertTrue(np.allclose(q_learner.q.get_q(1), 0.0))

        # test returned action
        self.assertEqual(action, 'L')
        self.assertIsNone(q_learner.perceive(0, 0.0, ['L', 'R'], reached_goal_state=True))
        self.assertIsNone(q_learner.perceive(0, 0.0, ['L', 'R'], episode_end=True))

        # force greedy action pick
        # we never pick random action if learning mode is off. since here we want to test, if greed_eps=0.0 leads to
        # picking the greedy action, we need to turn learning on.
        q_learner.set_learning_mode(learning_on=True)
        q_learner.greed_eps.eps = 0.0
        qf.q_table[0, 1] = 1.0
        action = q_learner.perceive(0, 1.0, ['L', 'R'])
        self.assertEqual(action, 'R')
        q_learner.greed_eps.eps = 0.2
        self.assertEqual(q_learner.last_state, 0)
        self.assertEqual(q_learner.last_action, 'R')
        self.assertEqual(q_learner.last_reward, 1.0)

        # test reset
        q_learner.reset()
        self.assertIsNone(q_learner.last_state)
        self.assertIsNone(q_learner.last_action)
        self.assertEqual(q_learner.last_reward, 0.0)

        q_learner.set_learning_mode(learning_on=True)
        qf.q_table = np.zeros((2, 2))
        # test update q
        action = q_learner.perceive(0, -0.2, ['L'], reached_goal_state=False, episode_end=False)
        self.assertEqual(action, 'L')
        # there should be no updates
        self.assertTrue(np.allclose(q_learner.q.get_q(0), 0.0))
        self.assertTrue(np.allclose(q_learner.q.get_q(1), 0.0))
        action = q_learner.perceive(0, -0.2, ['R'], reached_goal_state=False, episode_end=False)
        self.assertEqual(action, 'R')
        self.assertAlmostEqual(q_learner.q.get_q(0, 'L'), -0.02)
        self.assertAlmostEqual(q_learner.q.get_q(0, 'R'), 0.0)
        self.assertAlmostEqual(q_learner.q.get_q(1, 'L'), 0.0)
        self.assertAlmostEqual(q_learner.q.get_q(1, 'R'), 0.0)
        action = q_learner.perceive(1, 1.0, ['L', 'R'], reached_goal_state=True, episode_end=False)
        self.assertAlmostEqual(q_learner.q.get_q(1, 'L'), 0.1)
        self.assertAlmostEqual(q_learner.q.get_q(1, 'R'), 0.1)
        self.assertAlmostEqual(q_learner.q.get_q(0, 'L'), -0.02)
        self.assertAlmostEqual(q_learner.q.get_q(0, 'R'), -0.011)
        self.assertIsNone(action)
        self.assertEqual(q_learner.episodes_experienced, 1)

        # test episode end
        qf.q_table = np.zeros((2, 2))
        action = q_learner.perceive(0, -0.2, ['R'], reached_goal_state=False, episode_end=False)
        self.assertEqual(action, 'R')
        action = q_learner.perceive(1, 1.0, ['L', 'R'], reached_goal_state=False, episode_end=True)
        self.assertIsNone(action)
        # q values for state 1 should stay the same
        self.assertAlmostEqual(q_learner.q.get_q(0, 'L'), 0.0)
        self.assertAlmostEqual(q_learner.q.get_q(0, 'R'), -0.02)
        self.assertAlmostEqual(q_learner.q.get_q(1, 'L'), 0.0)
        self.assertAlmostEqual(q_learner.q.get_q(1, 'R'), 0.0)
        self.assertEqual(q_learner.episodes_experienced, 2)

    def run_q_learning_1(self, q_function):
        q_learner = QLearningAgent(q_function, discount_factor=1.0, greed_eps=self.eps)
        for i in range(10000):
            self.e_fin.run(q_learner, episode_length=np.inf)
        # expected q: [[0.6, 0.8], [1.0, 1.0]]
        expected_q = np.array([[0.6, 0.8], [1.0, 1.0]])
        q = np.zeros((2, 2))
        q[0] = q_function.get_q(0)
        q[1] = q_function.get_q(1)
        self.assertLess(np.sum(np.abs(q - expected_q)), TOL, msg="{0:s}\n{1:s}".format(q, expected_q))

        q_learner = QLearningAgent(q_function, discount_factor=0.9, greed_eps=self.eps)
        for i in range(10000):
            self.e_fin.run(q_learner, episode_length=np.inf)
        # expected q: [[0.43, 0.7], [1.0, 1.0]]
        expected_q = np.array([[0.43, 0.7], [1.0, 1.0]])
        q = np.zeros((2, 2))
        q[0] = q_function.get_q(0)
        q[1] = q_function.get_q(1)
        self.assertLess(np.sum(np.abs(q - expected_q)), TOL, msg="{0:s}\n{1:s}".format(q, expected_q))

    def run_q_learning_2(self, q_function):
        q_learner = QLearningAgent(q_function, discount_factor=0.9, greed_eps=self.eps)
        for i in range(1000):
            self.e_inf.run(q_learner, episode_length=100)
        # expected q: [[7.72, 8.8], [8.92, 10.0]]
        expected_q = np.array([[7.72, 8.8], [8.92, 10.0]])
        q = np.zeros((2, 2))
        q[0] = q_function.get_q(0)
        q[1] = q_function.get_q(1)
        self.assertLess(np.sum(np.abs(q - expected_q)), TOL, msg="{0:s}\n{1:s}".format(q, expected_q))

    @unittest.skipIf('--skipslow' in sys.argv, "Slow tests are turned off.")
    def test_q_learning_with_table(self):
        qf = QTableLookup(self.e_fin.state_space, self.a, learning_rate=0.01)
        self.run_q_learning_1(qf)
        qf = QTableLookup(self.e_inf.state_space, self.a, learning_rate=0.01)
        self.run_q_learning_2(qf)

    @unittest.skipIf('--skipslow' in sys.argv, "Slow tests are turned off.")
    def test_q_learning_with_nn(self):
        # a single layer network should suffice for this simple problem
        qf = QNeuralNetwork([], self.e_fin.state_space, self.a, learning_rate=0.01)
        self.run_q_learning_1(qf)
        qf = QNeuralNetwork([], self.e_inf.state_space, self.a, learning_rate=0.01)
        self.run_q_learning_2(qf)

