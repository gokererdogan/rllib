"""
rllib - Reinforcement Learning Library

Q-learning. Provides functionality for implementing Q-functions with tables or neural networks.
Methods here are applicable only to discrete action spaces. State space can be continuous or discrete.

Goker Erdogan
https://github.com/gokererdogan
"""

import numpy as np
import theano
import theano.tensor as T
import lasagne

from agent import Agent


class QFunction(object):
    """
    QFunction base class. This abstract class specifies the interface expected of Q-functions. A Q-function is function
    that maps state, action pairs to values (e.g., expected reward). Note that only finite action spaces are allowed.
    """
    def __init__(self, state_space, action_space):
        """
        Parameters:
            state_space (StateSpace)
            action_space (ActionSpace)
        """
        self.state_space = state_space
        self.action_space = action_space
        self.action_count = len(action_space)

    def get_q(self, state, action=None):
        """
        Return Q-values for state, action pair. If action is None, Q-values for all actions are returned.

        Parameters:
            state
            action

        Returns:
            float or list: Q-value for state, action pair. If action is None, Q-values of state for all actions.
        """
        raise NotImplementedError()

    def update_q(self, value, state, action=None):
        """
        Update Q-value of state, action pair towards value. If action is None, all Q-values for state are updated.

        Parameters:
            value (float): Target Q-value
            state
            action
        """
        raise NotImplementedError()


class QTableLookup(QFunction):
    """
    QTableLookup class. This class uses a table of state, action pairs to implement a Q-function. This is possible only
    for discrete state and action spaces.
    """
    def __init__(self, state_space, action_space, learning_rate):
        """
        Parameters:
            state_space (StateSpace)
            action_space (ActionSpace)
            learning_rate (float): Learning rate used in Q-value updates.
        """
        QFunction.__init__(self, state_space, action_space)
        self.state_count = len(state_space)
        self.learning_rate = learning_rate

        # initialize Q-function randomly
        self.q_table = np.random.rand(self.state_count, self.action_count) * 0.01

    def get_q(self, state, action=None):
        s_id = self.state_space.index(state)
        if action is None:
            return self.q_table[s_id, :]
        else:
            a_id = self.action_space.index(action)
            return self.q_table[s_id, a_id]

    def update_q(self, value, state, action=None):
        s_id = self.state_space.index(state)
        if action is None:
            # move current Q-values for state towards provided value.
            self.q_table[s_id, :] += self.learning_rate * (value - self.q_table[s_id, :])
        else:
            a_id = self.action_space.index(action)
            # move current Q-value for state, action pair towards provided value.
            self.q_table[s_id, a_id] += self.learning_rate * (value - self.q_table[s_id, a_id])

    def __str__(self):
        return str(self.q_table)


class QNeuralNetwork(QFunction):
    """
    QNeuralNetwork class. This class uses a neural network as a function approximator to implement a Q-function. Input
    to the neural network is a vector representation of the state and outputs are Q-values for each action. Note that
    the action space here is assumed to be discrete.
    """
    def __init__(self, neural_network, state_space, action_space, learning_rate):
        """
        Parameters:
            neural_network (list or lasagne.layers.Layer): Either a list containing the number of hidden units
                in each hidden layer or a lasagne.layers.Layer instance with input and output layers with the proper
                sizes and names input and output respectively.
            state_space (StateSpace)
            action_space (ActionSpace)
            learning_rate (float): Learning rate
        """
        QFunction.__init__(self, state_space, action_space)
        self.learning_rate = learning_rate

        # are we given a lasagne neural network?
        if isinstance(neural_network, lasagne.layers.Layer):
            self.nn = neural_network
            # find the input layer and get the input variable
            l = self.nn
            while True:
                if isinstance(l, lasagne.layers.InputLayer):
                    self.input = l.input_var
                    break
                l = l.input_layer
        elif isinstance(neural_network, list):  # are we given a list of hidden unit counts
            input_dim = self.state_space.to_vector(self.state_space.get_initial_state()).shape
            nn = lasagne.layers.InputLayer(shape=(1,) + input_dim)
            self.input = nn.input_var
            for i, n in enumerate(neural_network):
                nn = lasagne.layers.DenseLayer(incoming=nn, num_units=n)

            nn = lasagne.layers.DenseLayer(incoming=nn, num_units=self.action_count, nonlinearity=None)
            self.nn = nn
        else:
            raise ValueError("neural network has to be either a list or lasagne.layers.Layer instance.")

        # theano code for training
        self.target = T.vector('target')
        self.output = lasagne.layers.get_output(self.nn)
        self.forward = theano.function([self.input], self.output)

        # loss is squared error, (q_current - q_expected)**2
        self.loss = lasagne.objectives.squared_error(self.output, self.target)
        self.loss = lasagne.objectives.aggregate(self.loss)

        self.params = lasagne.layers.get_all_params(self.nn, trainable=True)
        self.updates = lasagne.updates.sgd(self.loss, self.params, learning_rate=self.learning_rate)

        self.train_function = theano.function([self.input, self.target], self.loss, updates=self.updates)

    def get_q(self, state, action=None):
        # get input size
        x = self.state_space.to_vector(state).astype(theano.config.floatX)
        x = x[np.newaxis, :]
        # run forward to get q-values for state
        q = self.forward(x)
        if action is None:
            return q[0]
        else:
            a_id = self.action_space.index(action)
            return q[0, a_id]

    def update_q(self, value, state, action=None):
        x = self.state_space.to_vector(state).astype(theano.config.floatX)
        x = x[np.newaxis, :]

        if action is None:
            # update all output units
            y = np.ones(self.action_count, dtype=theano.config.floatX) * value
        else:
            # update only the output unit for action
            # we use the old q-values for all other actions to prevent updating them.
            a_id = self.action_space.index(action)
            y = self.forward(x)[0]
            y[a_id] = value

        self.train_function(x, y)


class QLearningAgent(Agent):
    """
    QLearningAgent class. This class implements an agent that uses epsilon greedy Q-learning to learn the optimal policy.
    We can plug-in our choice of Q-function implementation (e.g., table lookup or neural network).
    """
    def __init__(self, q_function, discount_factor, greed_eps):
        """
        Parameters:
            q_function (QFunction)
            discount_factor (float): Reward discount factor
            greed_eps (ParameterSchedule): Schedule for epsilon of epsilon-greedy action picking strategy (probability
                of picking a random (rather than the greedy) action.
        """
        Agent.__init__(self, action_space=q_function.action_space)
        if discount_factor < 0.0 or discount_factor > 1.0:
            raise ValueError("Discount factor should be between 0.0 and 1.0.")
        self.discount_factor = discount_factor
        self.greed_eps = greed_eps
        # keep track of episodes experienced (this is for example used by parameter schedules)
        self.episodes_experienced = 0

        self.q = q_function

        # We need to keep track of last state, action, and reward
        self.last_state = None
        self.last_action = None
        self.last_reward = 0.0

    def reset(self):
        # at each episode start, this method is called to reset last state, action, and reward.
        self.last_state = None
        self.last_action = None
        self.last_reward = 0.0

    def perceive(self, state, reward, available_actions, reached_goal_state=False, episode_end=False):
        """
        Perceive and act. Uses Q-learning to update Q-values and picks the action using epsilon greedy policy.

        Parameters:
            state
            reward (float)
            available_actions (list): List of available action in state
            reached_goal_state (bool)
            episode_end (bool)
        """
        # perceive
        if self.learning_on:
            if reached_goal_state or episode_end:
                self.episodes_experienced += 1
                if reached_goal_state:
                    # update q-value for state (for all actions)
                    self.q.update_q(reward, state, None)

            if self.last_action is not None:  # if not on first trial
                # calculate expected Q-value
                estimated_q_sa = self.last_reward + self.discount_factor * (np.max(self.q.get_q(state, None)))
                # update Q-value for state action pair
                self.q.update_q(estimated_q_sa, self.last_state, self.last_action)

        # act
        if reached_goal_state or episode_end:
            action = None
        else:
            if not self.learning_on or np.random.rand() > self.greed_eps.get_value(self):
                # pick greedy action
                available_action_ids = [self.action_space.index(a) for a in available_actions]
                # mask unavailable actions
                q_s = self.q.get_q(state, None)
                mask_a = [i not in available_action_ids for i in range(len(q_s))]
                q_ma = np.ma.masked_array(q_s, mask=mask_a)
                action = self.action_space[q_ma.argmax()]
            else:
                # pick random action
                a_id = np.random.choice(len(available_actions))
                action = available_actions[a_id]

        self.last_reward = reward
        self.last_state = state
        self.last_action = action
        return action


