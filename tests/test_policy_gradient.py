"""
rllib - Reinforcement Learning Library

Unit tests for policy_gradient module.

Goker Erdogan
https://github.com/gokererdogan/
"""
import sys
import numpy as np
import unittest

from lasagne.updates import sgd
import lasagne.layers as ll

from rllib.parameter_schedule import GreedyEpsilonConstantSchedule
from rllib.examples.two_state_world import TwoStateFiniteWorldEnvironment, TwoStateInfiniteWorldEnvironment, \
    TwoStateActionSpace
from rllib.examples.simple_infinite_world import SimpleInfiniteWorldEnvironment, SimpleInfiniteWorldActionSpace
from rllib.policy_gradient import *


TOL = 1e-1  # equality tolerance


class TestPolicyNeuralNetworkMultinomial(unittest.TestCase):
    def setUp(self):
        self.e = TwoStateFiniteWorldEnvironment()
        self.a = TwoStateActionSpace()
        self.pnn = PolicyNeuralNetworkMultinomial([], self.e.state_space, self.a, learning_rate=1.0, optimizer=sgd)

    def test_init(self):
        # we shouldn't be able to create a PolicyNeuralNetwork instance because some required attributes are defined in
        # subclasses.
        self.assertRaises(AttributeError, PolicyNeuralNetwork, self.pnn.nn, self.e.state_space, self.a, 1.0, sgd)

        # neural network should have the right input shape
        nn = ll.InputLayer(shape=(1, 4))
        nn = ll.DenseLayer(nn, 2)
        self.assertRaises(ValueError, PolicyNeuralNetwork, nn, self.e.state_space, self.a, 1.0, sgd)

        # neural network should have as many output units as there are actions.
        nn = ll.InputLayer(shape=(1, 2))
        nn = ll.DenseLayer(nn, 4)
        self.assertRaises(ValueError, PolicyNeuralNetworkMultinomial, nn, self.e.state_space, self.a, 1.0, sgd)

        #
        self.assertAlmostEqual(self.pnn.learning_rate.get_value(), 1.0)
        self.assertEqual(len(self.pnn.params), 2)
        self.assertTrue(self.pnn.nn.W.get_value().shape, (2, 2))
        self.assertTrue(self.pnn.nn.b.get_value().shape, (2,))
        self.assertAlmostEqual(self.pnn.episode_reward.get_value(), 0.0)
        self.assertTrue(np.allclose(self.pnn.episode_grads[0].get_value(),
                                    np.zeros_like(self.pnn.params[0].get_value())))
        self.assertTrue(np.allclose(self.pnn.episode_grads[1].get_value(),
                                    np.zeros_like(self.pnn.params[1].get_value())))
        self.assertTrue(np.allclose(self.pnn.total_grads[0].get_value(),
                                    np.zeros_like(self.pnn.params[0].get_value())))
        self.assertTrue(np.allclose(self.pnn.total_grads[1].get_value(),
                                    np.zeros_like(self.pnn.params[1].get_value())))

    def test_get_action(self):
        # test forward
        self.pnn.nn.W.set_value(np.array([[0.1, 0.2], [0.3, 0.4]], dtype=theano.config.floatX))
        self.pnn.nn.b.set_value(np.array([-0.1, 0.1], dtype=theano.config.floatX))
        probs = self.pnn._forward([[1.0, 0.0]])[0]
        self.assertAlmostEqual(probs[0], np.exp(0.0) / (np.exp(0.0) + np.exp(0.3)))
        self.assertAlmostEqual(probs[1], np.exp(0.3) / (np.exp(0.0) + np.exp(0.3)))
        probs = self.pnn._forward([[0.0, 1.0]])[0]
        self.assertAlmostEqual(probs[0], np.exp(0.2) / (np.exp(0.2) + np.exp(0.5)))
        self.assertAlmostEqual(probs[1], np.exp(0.5) / (np.exp(0.2) + np.exp(0.5)))
        # test restricting available actions
        a = self.pnn.get_action(0, available_actions=['L'])
        self.assertEqual(a, 'L')
        a = self.pnn.get_action(1, available_actions=['R'])
        self.assertEqual(a, 'R')
        #
        a = self.pnn.get_action(0, available_actions=['L', 'R'])
        self.assertIn(a, ['L', 'R'])
        # make one action very high probability
        self.pnn.nn.W.set_value(np.array([[1000., 0.0], [0.0, 1000.]], dtype=theano.config.floatX))
        a = self.pnn.get_action(0, available_actions=['L', 'R'])
        self.assertEqual(a, 'L')
        a = self.pnn.get_action(1, available_actions=['L', 'R'])
        self.assertEqual(a, 'R')

    def test_accumulate_reward(self):
        self.pnn.accumulate_reward(1.0)
        self.assertAlmostEqual(self.pnn.episode_reward.get_value(), 1.0)
        self.pnn.accumulate_reward(2.0)
        self.assertAlmostEqual(self.pnn.episode_reward.get_value(), 3.0)
        self.pnn.episode_reward.set_value(0.0)
        self.assertAlmostEqual(self.pnn.episode_reward.get_value(), 0.0)
        self.pnn.accumulate_reward(2.0)
        self.assertAlmostEqual(self.pnn.episode_reward.get_value(), 2.0)

    def test_accumulate_gradients(self):
        self.pnn.nn.W.set_value(np.array([[0.1, 0.2], [0.3, 0.4]], dtype=theano.config.floatX))
        self.pnn.nn.b.set_value(np.array([-0.1, 0.1], dtype=theano.config.floatX))
        self.pnn.accumulate_gradients(0, 'L')
        pl1 = np.exp(0.0) / (np.exp(0.0) + np.exp(0.3))
        pr1 = np.exp(0.3) / (np.exp(0.0) + np.exp(0.3))
        dw = self.pnn.episode_grads[0].get_value()
        db = self.pnn.episode_grads[1].get_value()
        expected_dw = np.array([[(1-pl1), -pr1], [0.0, 0.0]])
        expected_db = np.array([(1-pl1), -pr1])
        self.assertTrue(np.allclose(dw, expected_dw),
                        msg="{0:s}\n{1:s}".format(dw, expected_dw))
        self.assertTrue(np.allclose(db, expected_db),
                        msg="{0:s}\n{1:s}".format(db, expected_db))
        self.pnn.accumulate_gradients(1, 'R')
        pl2 = np.exp(0.2) / (np.exp(0.2) + np.exp(0.5))
        pr2 = np.exp(0.5) / (np.exp(0.2) + np.exp(0.5))
        dw = self.pnn.episode_grads[0].get_value()
        db = self.pnn.episode_grads[1].get_value()
        expected_dw = np.array([[(1-pl1), -pr1], [-pl2, (1-pr2)]])
        expected_db = np.array([(1-pl1) - pl2, -pr1 + (1-pr2)])
        self.assertTrue(np.allclose(dw, expected_dw),
                        msg="{0:s}\n{1:s}".format(dw, expected_dw))
        self.assertTrue(np.allclose(db, expected_db),
                        msg="{0:s}\n{1:s}".format(db, expected_db))
        self.pnn.accumulate_gradients(0, 'R')
        pl3 = np.exp(0.0) / (np.exp(0.0) + np.exp(0.3))
        pr3 = np.exp(0.3) / (np.exp(0.0) + np.exp(0.3))
        dw = self.pnn.episode_grads[0].get_value()
        db = self.pnn.episode_grads[1].get_value()
        expected_dw = np.array([[(1-pl1) - pl3, -pr1 + (1-pr3)], [-pl2, (1-pr2)]])
        expected_db = np.array([(1-pl1) - pl2 - pl3, -pr1 + (1-pr2) + (1-pr3)])
        self.assertTrue(np.allclose(dw, expected_dw),
                        msg="{0:s}\n{1:s}".format(dw, expected_dw))
        self.assertTrue(np.allclose(db, expected_db),
                        msg="{0:s}\n{1:s}".format(db, expected_db))

    def test_end_episode(self):
        self.pnn.nn.W.set_value(np.array([[0.1, 0.2], [0.3, 0.4]], dtype=theano.config.floatX))
        self.pnn.nn.b.set_value(np.array([-0.1, 0.1], dtype=theano.config.floatX))
        # everything should be zero
        self.pnn.end_episode()
        self.assertAlmostEqual(self.pnn.episode_reward.get_value(), 0.0)
        self.assertTrue(np.allclose(self.pnn.episode_grads[0].get_value(),
                                    np.zeros_like(self.pnn.params[0].get_value())))
        self.assertTrue(np.allclose(self.pnn.episode_grads[1].get_value(),
                                    np.zeros_like(self.pnn.params[1].get_value())))
        self.assertTrue(np.allclose(self.pnn.total_grads[0].get_value(),
                                    np.zeros_like(self.pnn.params[0].get_value())))
        self.assertTrue(np.allclose(self.pnn.total_grads[1].get_value(),
                                    np.zeros_like(self.pnn.params[1].get_value())))
        #
        self.pnn.accumulate_reward(2.0)
        self.pnn.accumulate_gradients(0, 'L')
        pl1 = np.exp(0.0) / (np.exp(0.0) + np.exp(0.3))
        pr1 = np.exp(0.3) / (np.exp(0.0) + np.exp(0.3))
        expected_tdw = 2.0 * np.array([[(1-pl1), -pr1], [0.0, 0.0]])
        expected_tdb = 2.0 * np.array([(1-pl1), -pr1])
        self.assertAlmostEqual(self.pnn.episode_reward.get_value(), 2.0)
        self.assertTrue(np.allclose(self.pnn.total_grads[0].get_value(),
                                    np.zeros_like(self.pnn.params[0].get_value())))
        self.assertTrue(np.allclose(self.pnn.total_grads[1].get_value(),
                                    np.zeros_like(self.pnn.params[1].get_value())))
        # end episode and check totals
        self.pnn.end_episode()
        tdw = self.pnn.total_grads[0].get_value()
        tdb = self.pnn.total_grads[1].get_value()
        self.assertTrue(np.allclose(tdw, expected_tdw),
                        msg="{0:s}\n{1:s}".format(tdw, expected_tdw))
        self.assertTrue(np.allclose(tdb, expected_tdb),
                        msg="{0:s}\n{1:s}".format(tdb, expected_tdb))
        self.assertTrue(np.allclose(self.pnn.episode_grads[0].get_value(),
                                    np.zeros_like(self.pnn.params[0].get_value())))
        self.assertTrue(np.allclose(self.pnn.episode_grads[1].get_value(),
                                    np.zeros_like(self.pnn.params[1].get_value())))
        self.assertAlmostEqual(self.pnn.episode_reward.get_value(), 0.0)
        #
        self.pnn.accumulate_reward(1.0)
        self.pnn.accumulate_gradients(1, 'R')
        pl2 = np.exp(0.2) / (np.exp(0.2) + np.exp(0.5))
        pr2 = np.exp(0.5) / (np.exp(0.2) + np.exp(0.5))
        tdw = self.pnn.total_grads[0].get_value()
        tdb = self.pnn.total_grads[1].get_value()

        self.assertTrue(np.allclose(tdw, expected_tdw),
                        msg="{0:s}\n{1:s}".format(tdw, expected_tdw))
        self.assertTrue(np.allclose(tdb, expected_tdb),
                        msg="{0:s}\n{1:s}".format(tdb, expected_tdb))
        self.assertAlmostEqual(self.pnn.episode_reward.get_value(), 1.0)
        #
        self.pnn.accumulate_reward(0.5)
        self.pnn.accumulate_gradients(0, 'R')
        pl3 = np.exp(0.0) / (np.exp(0.0) + np.exp(0.3))
        pr3 = np.exp(0.3) / (np.exp(0.0) + np.exp(0.3))
        tdw = self.pnn.total_grads[0].get_value()
        tdb = self.pnn.total_grads[1].get_value()

        self.assertTrue(np.allclose(tdw, expected_tdw),
                        msg="{0:s}\n{1:s}".format(tdw, expected_tdw))
        self.assertTrue(np.allclose(tdb, expected_tdb),
                        msg="{0:s}\n{1:s}".format(tdb, expected_tdb))
        self.assertAlmostEqual(self.pnn.episode_reward.get_value(), 1.5)
        #
        self.pnn.end_episode()
        tdw = self.pnn.total_grads[0].get_value()
        tdb = self.pnn.total_grads[1].get_value()

        expected_tdw += (1.5 * np.array([[-pl3, (1-pr3)], [-pl2, (1-pr2)]]))
        expected_tdb += (1.5 * np.array([-pl2-pl3, (1-pr2) + (1-pr3)]))
        self.assertTrue(np.allclose(tdw, expected_tdw),
                        msg="{0:s}\n{1:s}".format(tdw, expected_tdw))
        self.assertTrue(np.allclose(tdb, expected_tdb),
                        msg="{0:s}\n{1:s}".format(tdb, expected_tdb))
        self.assertTrue(np.allclose(self.pnn.episode_grads[0].get_value(),
                                    np.zeros_like(self.pnn.params[0].get_value())))
        self.assertTrue(np.allclose(self.pnn.episode_grads[1].get_value(),
                                    np.zeros_like(self.pnn.params[1].get_value())))
        self.assertAlmostEqual(self.pnn.episode_reward.get_value(), 0.0)

    def test_update_params(self):
        w = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=theano.config.floatX)
        b = np.array([-0.1, 0.1], dtype=theano.config.floatX)
        self.pnn.nn.W.set_value(w)
        self.pnn.nn.b.set_value(b)

        # cannot update if no gradients are accumulated
        self.assertRaises(RuntimeError, self.pnn.update_parameters)
        # params shouldn't change
        self.assertTrue(np.allclose(self.pnn.nn.W.get_value(), w))
        self.assertTrue(np.allclose(self.pnn.nn.b.get_value(), b))
        self.assertTrue(np.allclose(self.pnn.total_grads[0].get_value(),
                                    np.zeros_like(self.pnn.params[0].get_value())))
        self.assertTrue(np.allclose(self.pnn.total_grads[1].get_value(),
                                    np.zeros_like(self.pnn.params[1].get_value())))
        #
        self.pnn.accumulate_reward(2.0)
        self.pnn.accumulate_gradients(0, 'L')
        pl1 = np.exp(0.0) / (np.exp(0.0) + np.exp(0.3))
        pr1 = np.exp(0.3) / (np.exp(0.0) + np.exp(0.3))
        # update_params without first calling end episode shouldn't be possible
        self.assertRaises(RuntimeError, self.pnn.update_parameters)

        self.assertTrue(np.allclose(self.pnn.nn.W.get_value(), w))
        self.assertTrue(np.allclose(self.pnn.nn.b.get_value(), b))
        # end episode, update params and check new values
        self.pnn.end_episode()
        self.pnn.update_parameters()
        expected_tdw = 2.0 * np.array([[(1-pl1), -pr1], [0.0, 0.0]])
        expected_tdb = 2.0 * np.array([(1-pl1), -pr1])

        self.assertTrue(np.allclose(self.pnn.nn.W.get_value(), w + expected_tdw))
        self.assertTrue(np.allclose(self.pnn.nn.b.get_value(), b + expected_tdb))
        self.assertTrue(np.allclose(self.pnn.total_grads[0].get_value(),
                                    np.zeros_like(self.pnn.params[0].get_value())))
        self.assertTrue(np.allclose(self.pnn.total_grads[1].get_value(),
                                    np.zeros_like(self.pnn.params[1].get_value())))


class TestPolicyNeuralNetworkNormal(unittest.TestCase):
    def setUp(self):
        self.e = SimpleInfiniteWorldEnvironment()
        self.a = SimpleInfiniteWorldActionSpace()
        self.pnn = PolicyNeuralNetworkNormal([], self.e.state_space, self.a, learning_rate=1.0, optimizer=sgd,
                                             cov_type='diagonal')

    def test_init(self):
        # if cov_type is diagonal neural network should have 2 times as many output units as action dimension.
        nn = ll.InputLayer(shape=(1, 1))
        nn = ll.DenseLayer(nn, 4)
        self.assertRaises(ValueError, PolicyNeuralNetworkNormal, nn, self.e.state_space, self.a, 1.0, sgd, 'diagonal')
        # if cov_type is identity neural network should have as many output units as action dimension.
        self.assertRaises(ValueError, PolicyNeuralNetworkNormal, nn, self.e.state_space, self.a, 1.0, sgd, 'identity')

        # cov_type can be identity or diagonal
        self.assertRaises(ValueError, PolicyNeuralNetworkNormal, [], self.e.state_space, self.a, 1.0, sgd, 'xxx')

        #
        self.assertEqual(self.pnn.cov_type, 'diagonal')
        self.assertAlmostEqual(self.pnn.learning_rate.get_value(), 1.0)
        self.assertEqual(len(self.pnn.params), 2)
        self.assertTrue(self.pnn.nn.W.get_value().shape, (1, 2))
        self.assertTrue(self.pnn.nn.b.get_value().shape, (2,))
        self.assertAlmostEqual(self.pnn.episode_reward.get_value(), 0.0)
        self.assertTrue(np.allclose(self.pnn.episode_grads[0].get_value(),
                                    np.zeros_like(self.pnn.params[0].get_value())))
        self.assertTrue(np.allclose(self.pnn.episode_grads[1].get_value(),
                                    np.zeros_like(self.pnn.params[1].get_value())))
        self.assertTrue(np.allclose(self.pnn.total_grads[0].get_value(),
                                    np.zeros_like(self.pnn.params[0].get_value())))
        self.assertTrue(np.allclose(self.pnn.total_grads[1].get_value(),
                                    np.zeros_like(self.pnn.params[1].get_value())))

    def test_get_action(self):
        # test forward
        self.pnn.nn.W.set_value(np.array([[0.1, 0.2]], dtype=theano.config.floatX))
        self.pnn.nn.b.set_value(np.array([-0.1, 0.1], dtype=theano.config.floatX))
        x = np.random.randn()
        out = self.pnn._forward([[x]])[0]
        self.assertAlmostEqual(out[0], 0.1*x-0.1)
        self.assertAlmostEqual(out[1], 0.2*x+0.1)
        # make one action very high probability
        self.pnn.nn.W.set_value(np.array([[1.0, -1000.]], dtype=theano.config.floatX))
        self.pnn.nn.b.set_value(np.array([0.0, 0.0], dtype=theano.config.floatX))
        x = np.abs(np.random.randn())
        a = self.pnn.get_action(np.array([x]), None)
        self.assertAlmostEqual(a, x)
        self.pnn.nn.W.set_value(np.array([[0.0, 0.0]], dtype=theano.config.floatX))
        self.pnn.nn.b.set_value(np.array([-1.0, -1000.0], dtype=theano.config.floatX))
        x = np.random.randn()
        a = self.pnn.get_action(np.array([x]), None)
        self.assertAlmostEqual(a, -1.0)

    def test_accumulate_gradients(self):
        wm, bm, ws, bs = np.random.randn(4)
        self.pnn.nn.W.set_value(np.array([[wm, ws]], dtype=theano.config.floatX))
        self.pnn.nn.b.set_value(np.array([bm, bs], dtype=theano.config.floatX))
        x = np.random.randn()
        a = np.random.randn()  # action
        self.pnn.selected_action.set_value(np.array([a]))
        m = x*wm + bm
        logs = x*ws + bs
        s = np.exp(logs)
        dwm = x * (a-m) / s**2
        dws = x * ((((a-m)/s)**2) - 1)
        dbm = (a-m) / s**2
        dbs = (((a-m)/s)**2) - 1
        # test log_p
        calc_log_p = theano.function([self.pnn.input], self.pnn.log_p)
        log_p = calc_log_p(np.array([[x]]))
        expected_log_p = -logs - 0.5*((a-m)/s)**2
        self.assertAlmostEqual(log_p, expected_log_p)
        # test accumulate gradient
        self.pnn.accumulate_gradients(np.array([x]), np.array([a]))
        dw = self.pnn.episode_grads[0].get_value()
        db = self.pnn.episode_grads[1].get_value()
        expected_dw = np.array([dwm, dws])
        expected_db = np.array([dbm, dbs])
        self.assertTrue(np.allclose(dw, expected_dw),
                        msg="{0:s}\n{1:s}".format(dw, expected_dw))
        self.assertTrue(np.allclose(db, expected_db),
                        msg="{0:s}\n{1:s}".format(db, expected_db))

    def test_end_episode(self):
        wm, bm, ws, bs = np.random.randn(4)
        self.pnn.nn.W.set_value(np.array([[wm, ws]], dtype=theano.config.floatX))
        self.pnn.nn.b.set_value(np.array([bm, bs], dtype=theano.config.floatX))

        # everything should be zero
        self.pnn.end_episode()
        self.assertAlmostEqual(self.pnn.episode_reward.get_value(), 0.0)
        self.assertTrue(np.allclose(self.pnn.episode_grads[0].get_value(),
                                    np.zeros_like(self.pnn.params[0].get_value())))
        self.assertTrue(np.allclose(self.pnn.episode_grads[1].get_value(),
                                    np.zeros_like(self.pnn.params[1].get_value())))
        self.assertTrue(np.allclose(self.pnn.total_grads[0].get_value(),
                                    np.zeros_like(self.pnn.params[0].get_value())))
        self.assertTrue(np.allclose(self.pnn.total_grads[1].get_value(),
                                    np.zeros_like(self.pnn.params[1].get_value())))
        #
        x = np.random.randn()
        a = np.random.randn()  # action
        m = x*wm + bm
        logs = x*ws + bs
        s = np.exp(logs)
        dwm = x * (a-m) / s**2
        dws = x * ((((a-m)/s)**2) - 1)
        dbm = (a-m) / s**2
        dbs = (((a-m)/s)**2) - 1
        expected_dw = np.array([dwm, dws])
        expected_db = np.array([dbm, dbs])

        self.pnn.accumulate_reward(2.0)
        self.pnn.accumulate_gradients(np.array([x]), np.array([a]))
        self.assertAlmostEqual(self.pnn.episode_reward.get_value(), 2.0)
        self.assertTrue(np.allclose(self.pnn.total_grads[0].get_value(),
                                    np.zeros_like(self.pnn.params[0].get_value())))
        self.assertTrue(np.allclose(self.pnn.total_grads[1].get_value(),
                                    np.zeros_like(self.pnn.params[1].get_value())))
        # end episode and check totals
        self.pnn.end_episode()
        tdw = self.pnn.total_grads[0].get_value()
        tdb = self.pnn.total_grads[1].get_value()
        expected_tdw = 2*expected_dw
        expected_tdb = 2*expected_db
        self.assertTrue(np.allclose(tdw, expected_tdw),
                        msg="{0:s}\n{1:s}".format(tdw, expected_tdw))
        self.assertTrue(np.allclose(tdb, expected_tdb),
                        msg="{0:s}\n{1:s}".format(tdb, expected_tdb))
        self.assertTrue(np.allclose(self.pnn.episode_grads[0].get_value(),
                                    np.zeros_like(self.pnn.params[0].get_value())))
        self.assertTrue(np.allclose(self.pnn.episode_grads[1].get_value(),
                                    np.zeros_like(self.pnn.params[1].get_value())))
        self.assertAlmostEqual(self.pnn.episode_reward.get_value(), 0.0)

        #
        x = np.random.randn()
        a = np.random.randn()  # action
        m = x*wm + bm
        logs = x*ws + bs
        s = np.exp(logs)
        dwm = x * (a-m) / s**2
        dws = x * ((((a-m)/s)**2) - 1)
        dbm = (a-m) / s**2
        dbs = (((a-m)/s)**2) - 1
        expected_dw1 = np.array([dwm, dws])
        expected_db1 = np.array([dbm, dbs])

        self.pnn.accumulate_reward(-1.0)
        self.pnn.accumulate_gradients(np.array([x]), np.array([a]))
        tdw = self.pnn.total_grads[0].get_value()
        tdb = self.pnn.total_grads[1].get_value()

        self.assertTrue(np.allclose(tdw, expected_tdw),
                        msg="{0:s}\n{1:s}".format(tdw, expected_tdw))
        self.assertTrue(np.allclose(tdb, expected_tdb),
                        msg="{0:s}\n{1:s}".format(tdb, expected_tdb))
        self.assertAlmostEqual(self.pnn.episode_reward.get_value(), -1.0)
        #
        x = np.random.randn()
        a = np.random.randn()  # action
        m = x*wm + bm
        logs = x*ws + bs
        s = np.exp(logs)
        dwm = x * (a-m) / s**2
        dws = x * ((((a-m)/s)**2) - 1)
        dbm = (a-m) / s**2
        dbs = (((a-m)/s)**2) - 1
        expected_dw2 = np.array([dwm, dws])
        expected_db2 = np.array([dbm, dbs])

        self.pnn.accumulate_reward(0.5)
        self.pnn.accumulate_gradients(np.array([x]), np.array([a]))
        tdw = self.pnn.total_grads[0].get_value()
        tdb = self.pnn.total_grads[1].get_value()

        self.assertTrue(np.allclose(tdw, expected_tdw),
                        msg="{0:s}\n{1:s}".format(tdw, expected_tdw))
        self.assertTrue(np.allclose(tdb, expected_tdb),
                        msg="{0:s}\n{1:s}".format(tdb, expected_tdb))
        self.assertAlmostEqual(self.pnn.episode_reward.get_value(), -0.5)
        #
        self.pnn.end_episode()
        tdw = self.pnn.total_grads[0].get_value()
        tdb = self.pnn.total_grads[1].get_value()

        expected_tdw += -0.5*(expected_dw1 + expected_dw2)
        expected_tdb += -0.5*(expected_db1 + expected_db2)

        self.assertTrue(np.allclose(tdw, expected_tdw),
                        msg="{0:s}\n{1:s}".format(tdw, expected_tdw))
        self.assertTrue(np.allclose(tdb, expected_tdb),
                        msg="{0:s}\n{1:s}".format(tdb, expected_tdb))
        self.assertTrue(np.allclose(self.pnn.episode_grads[0].get_value(),
                                    np.zeros_like(self.pnn.params[0].get_value())))
        self.assertTrue(np.allclose(self.pnn.episode_grads[1].get_value(),
                                    np.zeros_like(self.pnn.params[1].get_value())))
        self.assertAlmostEqual(self.pnn.episode_reward.get_value(), 0.0)

    def test_update_params(self):
        wm, bm, ws, bs = np.random.randn(4)
        w = np.array([[wm, ws]]).astype(theano.config.floatX)
        b = np.array([bm, bs]).astype(theano.config.floatX)
        self.pnn.nn.W.set_value(w)
        self.pnn.nn.b.set_value(b)

        # cannot update if no gradients are accumulated
        self.assertRaises(RuntimeError, self.pnn.update_parameters)
        # params shouldn't change
        self.assertTrue(np.allclose(self.pnn.nn.W.get_value(), w))
        self.assertTrue(np.allclose(self.pnn.nn.b.get_value(), b))
        self.assertTrue(np.allclose(self.pnn.total_grads[0].get_value(),
                                    np.zeros_like(self.pnn.params[0].get_value())))
        self.assertTrue(np.allclose(self.pnn.total_grads[1].get_value(),
                                    np.zeros_like(self.pnn.params[1].get_value())))

        #
        x = np.random.randn()
        a = np.random.randn()  # action
        m = x*wm + bm
        logs = x*ws + bs
        s = np.exp(logs)
        dwm = x * (a-m) / s**2
        dws = x * ((((a-m)/s)**2) - 1)
        dbm = (a-m) / s**2
        dbs = (((a-m)/s)**2) - 1
        expected_dw1 = np.array([dwm, dws])
        expected_db1 = np.array([dbm, dbs])

        self.pnn.accumulate_reward(2.0)
        self.pnn.accumulate_gradients(np.array([x]), np.array([a]))

        # update_params without first calling end episode shouldn't be possible
        self.assertRaises(RuntimeError, self.pnn.update_parameters)

        self.assertTrue(np.allclose(self.pnn.nn.W.get_value(), w))
        self.assertTrue(np.allclose(self.pnn.nn.b.get_value(), b))

        # end episode
        self.pnn.end_episode()
        expected_dw1 *= 2.0
        expected_db1 *= 2.0
        self.assertTrue(np.allclose(self.pnn.episode_grads[0].get_value(), 0.0))
        self.assertTrue(np.allclose(self.pnn.episode_grads[1].get_value(), 0.0))
        self.assertTrue(np.allclose(self.pnn.total_grads[0].get_value(), expected_dw1))
        self.assertTrue(np.allclose(self.pnn.total_grads[1].get_value(), expected_db1))
        self.assertAlmostEqual(self.pnn.samples_collected.get_value(), 1.0)

        #
        x = np.random.randn()
        a = np.random.randn()  # action
        m = x*wm + bm
        logs = x*ws + bs
        s = np.exp(logs)
        dwm = x * (a-m) / s**2
        dws = x * ((((a-m)/s)**2) - 1)
        dbm = (a-m) / s**2
        dbs = (((a-m)/s)**2) - 1
        expected_dw2 = np.array([dwm, dws])
        expected_db2 = np.array([dbm, dbs])

        self.pnn.accumulate_reward(-0.5)
        self.pnn.accumulate_gradients(np.array([x]), np.array([a]))

        # end episode
        self.pnn.end_episode()
        expected_dw2 *= -0.5
        expected_db2 *= -0.5
        expected_tdw = expected_dw1 + expected_dw2
        expected_tdb = expected_db1 + expected_db2
        self.assertTrue(np.allclose(self.pnn.episode_grads[0].get_value(), 0.0))
        self.assertTrue(np.allclose(self.pnn.episode_grads[1].get_value(), 0.0))
        self.assertTrue(np.allclose(self.pnn.total_grads[0].get_value(), expected_tdw))
        self.assertTrue(np.allclose(self.pnn.total_grads[1].get_value(), expected_tdb))
        self.assertAlmostEqual(self.pnn.samples_collected.get_value(), 2.0)

        # update params and check new values
        self.pnn.update_parameters()
        self.assertTrue(np.allclose(self.pnn.nn.W.get_value(), w + expected_tdw / 2.0)) # divide by episodes_experienced
        self.assertTrue(np.allclose(self.pnn.nn.b.get_value(), b + expected_tdb / 2.0))
        self.assertTrue(np.allclose(self.pnn.total_grads[0].get_value(), 0.0))
        self.assertTrue(np.allclose(self.pnn.total_grads[1].get_value(), 0.0))
        self.assertAlmostEqual(self.pnn.samples_collected.get_value(), 0.0)


class TestPolicyGradientAgent(unittest.TestCase):
    def setUp(self):
        self.e_fin = TwoStateFiniteWorldEnvironment()
        self.e_inf = TwoStateInfiniteWorldEnvironment()
        self.a = TwoStateActionSpace()
        self.eps = GreedyEpsilonConstantSchedule(0.2)

    def test_init(self):
        pf = PolicyNeuralNetworkMultinomial([], self.e_fin.state_space, self.a, learning_rate=0.1, optimizer=sgd)
        self.assertRaises(ValueError, PolicyGradientAgent, pf, -0.1, self.eps, 5)
        self.assertRaises(ValueError, PolicyGradientAgent, pf, 1.5, self.eps, 5)
        pg_learner = PolicyGradientAgent(pf, discount_factor=0.9, greed_eps=self.eps, update_freq=5)
        self.assertEqual(pg_learner.discount_factor, 0.9)
        self.assertEqual(pg_learner.update_freq, 5)
        self.assertEqual(pg_learner.episodes_experienced, 0)
        self.assertEqual(pg_learner.trials_experienced, 0)

    def test_perceive(self):
        pf = PolicyNeuralNetworkMultinomial([], self.e_fin.state_space, self.a, learning_rate=0.1, optimizer=sgd)
        pf.nn.W.set_value(np.zeros((2, 2), dtype=theano.config.floatX))
        pf.nn.b.set_value(np.zeros((2,), dtype=theano.config.floatX))
        pg_learner = PolicyGradientAgent(pf, discount_factor=0.9, greed_eps=self.eps, update_freq=1)

        # test learning mode
        pg_learner.set_learning_mode(learning_on=False)
        _ = pg_learner.perceive(0, -0.2, ['L', 'R'])
        action = pg_learner.perceive(1, 1.0, ['L'])
        self.assertTrue(np.allclose(pf.nn.W.get_value(), 0.0))
        self.assertTrue(np.allclose(pf.nn.b.get_value(), 0.0))

        # test returned action
        self.assertEqual(action, 'L')
        self.assertIsNone(pg_learner.perceive(0, 0.0, ['L', 'R'], reached_goal_state=True))
        self.assertIsNone(pg_learner.perceive(0, 0.0, ['L', 'R'], episode_end=True))
        self.assertEqual(pg_learner.episodes_experienced, 0)

        # force greedy action pick
        pg_learner.set_learning_mode(learning_on=True)
        pg_learner.greed_eps.eps = 0.0
        pf.nn.W.set_value(np.array([[1000., 0.0], [0.0, 1000.]], dtype=theano.config.floatX))
        action = pg_learner.perceive(0, -0.2, ['L', 'R'])
        self.assertEqual(action, 'L')
        self.assertEqual(pg_learner.trials_experienced, 1)
        self.assertEqual(pg_learner.episodes_experienced, 0)
        action = pg_learner.perceive(1, 1.0, ['L', 'R'])
        self.assertEqual(action, 'R')
        self.assertEqual(pg_learner.trials_experienced, 2)
        self.assertEqual(pg_learner.episodes_experienced, 0)
        action = pg_learner.perceive(1, 1.0, ['L', 'R'], reached_goal_state=True)
        self.assertEqual(pg_learner.trials_experienced, 0)
        self.assertEqual(pg_learner.episodes_experienced, 1)
        pg_learner.greed_eps.eps = 0.2

        # test update frequency parameter
        pg_learner.set_learning_mode(learning_on=True)
        pg_learner.episodes_experienced = 0
        pg_learner.trials_experienced = 0
        pg_learner.update_freq = 2

        w = np.random.randn(2, 2).astype(theano.config.floatX)
        b = np.random.randn(2).astype(theano.config.floatX)
        pf.nn.W.set_value(w)
        pf.nn.b.set_value(b)
        pg_learner.perceive(0, -0.2, ['L', 'R'], reached_goal_state=False, episode_end=False)
        pg_learner.perceive(1, 1.0, ['L', 'R'], reached_goal_state=True, episode_end=False)
        self.assertTrue(np.allclose(pf.nn.W.get_value(), w))
        self.assertTrue(np.allclose(pf.nn.b.get_value(), b))
        pg_learner.perceive(0, -0.2, ['L', 'R'], reached_goal_state=False, episode_end=False)
        pg_learner.perceive(1, 1.0, ['L', 'R'], reached_goal_state=True, episode_end=False)
        self.assertFalse(np.allclose(pf.nn.W.get_value(), w))
        self.assertFalse(np.allclose(pf.nn.b.get_value(), b))

        # test parameter updates
        pg_learner.episodes_experienced = 0
        pg_learner.trials_experienced = 0
        pg_learner.update_freq = 1

        w = np.random.randn(2, 2).astype(theano.config.floatX)
        b = np.random.randn(2).astype(theano.config.floatX)
        pf.nn.W.set_value(w)
        pf.nn.b.set_value(b)

        # get probabilities of actions
        prob_ls1, prob_rs1 = pf._forward([[1.0, 0.0]])[0]
        prob_ls2, prob_rs2 = pf._forward([[0.0, 1.0]])[0]

        action = pg_learner.perceive(0, -0.2, ['L'], reached_goal_state=False, episode_end=False)
        self.assertAlmostEqual(pf.episode_reward.get_value(), -0.2)
        self.assertEqual(action, 'L')
        action = pg_learner.perceive(0, -0.2, ['R'], reached_goal_state=False, episode_end=False)
        self.assertEqual(action, 'R')
        self.assertAlmostEqual(pf.episode_reward.get_value(), -0.38)
        action = pg_learner.perceive(1, 1.0, ['R'], reached_goal_state=False, episode_end=False)
        self.assertEqual(action, 'R')
        self.assertAlmostEqual(pf.episode_reward.get_value(), 0.43)
        action = pg_learner.perceive(1, 1.0, ['L', 'R'], reached_goal_state=False, episode_end=True)
        self.assertIsNone(action)
        self.assertEqual(pg_learner.episodes_experienced, 1)
        # check updated parameter values
        tot_reward = 1.159
        expected_w = w + (0.1 * tot_reward * np.array([[(1 - 2*prob_ls1), (1 - 2*prob_rs1)], [-prob_ls2, 1-prob_rs2]]))
        expected_b = b + (0.1 * tot_reward * np.array([(1 - 2*prob_ls1 - prob_ls2), (2 - 2*prob_rs1 - prob_rs2)]))
        uw = pf.nn.W.get_value()
        ub = pf.nn.b.get_value()
        self.assertTrue(np.allclose(uw, expected_w), msg="{0:s}\n{1:s}".format(uw, expected_w))
        self.assertTrue(np.allclose(ub, expected_b), msg="{0:s}\n{1:s}".format(ub, expected_b))

    @unittest.skipIf('--skipslow' in sys.argv, "Slow tests are turned off.")
    def test_policy_gradient(self):
        # test finite horizon
        pf = PolicyNeuralNetworkMultinomial([], self.e_fin.state_space, self.a, learning_rate=0.01, optimizer=sgd)
        pg_learner = PolicyGradientAgent(pf, discount_factor=1.0, greed_eps=self.eps, update_freq=1)
        for i in range(10000):
            self.e_fin.run(pg_learner, episode_length=np.inf)
        prob_s1 = pf._forward([[1.0, 0.0]])[0]
        prob_s2 = pf._forward([[0.0, 1.0]])[0]
        # all we can expect is that the agent prefers action R in state 0
        self.assertLess(prob_s1[0], prob_s1[1])

        # test infinite horizon
        pf = PolicyNeuralNetworkMultinomial([], self.e_inf.state_space, self.a, learning_rate=0.01, optimizer=sgd)
        pg_learner = PolicyGradientAgent(pf, discount_factor=0.9, greed_eps=self.eps, update_freq=1)
        for i in range(1000):
            # we need to keep the episodes short, because as the episodes get longer, the performance difference between
            # policies become smaller (hence we don't converge to any solution, or converge to one random policy)
            self.e_inf.run(pg_learner, episode_length=10)
        prob_s1 = pf._forward([[1.0, 0.0]])[0]
        prob_s2 = pf._forward([[0.0, 1.0]])[0]
        # all we can expect is that the agent prefers action R in both states
        self.assertLess(prob_s1[0], prob_s1[1])
        self.assertLess(prob_s2[0], prob_s2[1])



