"""
rllib - Reinforcement Learning Library

Unit tests for environment module.

Goker Erdogan
https://github.com/gokererdogan/
"""
import sys
import unittest
import numpy as np

from rllib.agent import Agent
from rllib.examples.two_state_world import TwoStateFiniteWorldEnvironment, \
    TwoStateInfiniteWorldEnvironment, TwoStateWorldAgent
from rllib.examples.two_state_mh_world import TwoStateHypothesis, TwoStateMHStateSpace, TwoStateMHActionSpace, \
    TwoStateMHEnvironment


class DummyTwoStateWorldMHAgent(Agent):
    def __init__(self, action_probs=(0.5, 0.5)):
        Agent.__init__(self, TwoStateMHActionSpace())
        self.action_probs = action_probs

    def get_action(self, state, available_actions=None):
        return np.random.choice(self.action_space, p=self.action_probs)

    def get_action_probability(self, state, action=None):
        if action == 'L':
            return self.action_probs[0]
        elif action == 'R':
            return self.action_probs[1]
        else:
            raise ValueError("Unknown action!")

    def perceive(self, state, reward, available_actions, reached_goal_state=False, episode_end=False):
        return self.get_action(state)

    def reset(self):
        pass


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


class TestMHEnvironment(unittest.TestCase):
    def test_init(self):
        ss = TwoStateMHStateSpace(reward_type='log_p')
        e = TwoStateMHEnvironment(ss)
        self.assertTrue(e.state_space is ss)

    def test_augment_state(self):
        ss = TwoStateMHStateSpace(reward_type='log_p')
        e = TwoStateMHEnvironment(ss)
        s = {'hypothesis': TwoStateHypothesis()}
        self.assertTrue(e._augment_state(s) is s)

    def test_advance(self):
        ss = TwoStateMHStateSpace(reward_type='log_p')
        e = TwoStateMHEnvironment(ss)
        a = DummyTwoStateWorldMHAgent()
        s = {'hypothesis': TwoStateHypothesis(state=0), 'is_accepted': True, 'log_p_increase': 0.0}
        e.current_state = s
        e.active_agent = a
        ns = e._advance('R')
        self.assertIn(ns['hypothesis'].state, [0, 1])
        self.assertIn(ns['is_accepted'], [False, True])
        if ns['hypothesis'].state == 0:
            self.assertAlmostEqual(ns['log_p_increase'], 0.0)
        else:
            self.assertAlmostEqual(ns['log_p_increase'], np.log(0.6) - np.log(0.4))

        # move left should not be accepted (very low probability)
        ns = e._advance('L')
        self.assertEqual(ns['hypothesis'].state, 0)
        self.assertEqual(ns['is_accepted'], False)
        self.assertAlmostEqual(ns['log_p_increase'], 0.0)

        #
        s = {'hypothesis': TwoStateHypothesis(state=1), 'is_accepted': True, 'log_p_increase': 0.0}
        e.current_state = s
        e.active_agent = a
        ns = e._advance('L')
        self.assertIn(ns['hypothesis'].state, [0, 1])
        self.assertIn(ns['is_accepted'], [False, True])
        if ns['hypothesis'].state == 0:
            self.assertAlmostEqual(ns['log_p_increase'], np.log(0.4) - np.log(0.6))
        else:
            self.assertAlmostEqual(ns['log_p_increase'], 0.0)

    def test_set_observed_data(self):
        ss = TwoStateMHStateSpace(reward_type='log_p')
        e = TwoStateMHEnvironment(ss)
        d = np.random.rand()
        e.set_observed_data(d)
        self.assertEqual(ss.data, d)
        d = np.random.rand(3, 2)
        e.set_observed_data(d)
        self.assertTrue(ss.data is d)

    @unittest.skipIf('--skipslow' in sys.argv, "Slow tests are turned off.")
    def test_mh_sampling(self):
        # test if MHEnvironment samples from the right distribution
        ss = TwoStateMHStateSpace(reward_type='log_p')
        e = TwoStateMHEnvironment(ss)
        a = DummyTwoStateWorldMHAgent()
        states, actions, rewards = e.run(a, episode_length=100000)
        # we expect to see 40% 0s, 60% 1s
        states = np.array([s['hypothesis'].state for s in states])
        p0 = np.mean(states == 0)
        p1 = np.mean(states == 1)
        self.assertAlmostEqual(p0, 0.4, places=1)
        self.assertAlmostEqual(p1, 0.6, places=1)

        # test with a different action policy
        a = DummyTwoStateWorldMHAgent(action_probs=[0.8, 0.2])
        states, actions, rewards = e.run(a, episode_length=100000)
        # we expect to see 40% 0s, 60% 1s
        states = np.array([s['hypothesis'].state for s in states])
        p0 = np.mean(states == 0)
        p1 = np.mean(states == 1)
        self.assertAlmostEqual(p0, 0.4, places=1)
        self.assertAlmostEqual(p1, 0.6, places=1)
