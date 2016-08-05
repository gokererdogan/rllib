import numpy as np
import theano
import theano.tensor as T
import lasagne

from agent import Agent


class QFunction(object):
    def __init__(self, state_space, action_space):
        self.state_space = state_space
        self.state_count = len(state_space)
        self.action_space = action_space
        self.action_count = len(action_space)

    def get_q(self, state, action=None):
        raise NotImplementedError()

    def update_q(self, value, state, action=None):
        raise NotImplementedError()


class QTableLookup(QFunction):
    def __init__(self, state_space, action_space, learning_rate):
        QFunction.__init__(self, state_space, action_space)
        self.learning_rate = learning_rate

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
            self.q_table[s_id, :] += self.learning_rate * (value - self.q_table[s_id, :])
        else:
            a_id = self.action_space.index(action)
            self.q_table[s_id, a_id] += self.learning_rate * (value - self.q_table[s_id, a_id])

    def __str__(self):
        return str(self.q_table)


class QNeuralNetwork(QFunction):
    def __init__(self, neural_network, state_space, action_space, learning_params):
        """
        Parameters:
            neural_network (list or lasagne.layers.Layer instance): Either a list containing the number of hidden units
                in each hidden layer or a lasagne.layers.Layer instance with input and output layers with the proper
                sizes and names input and output respectively.
            state_space (StateSpace instance):
            action_space (ActionSpace instance):
            learning_params (dict):
        """
        QFunction.__init__(self, state_space, action_space)
        self.learning_params = learning_params

        if isinstance(neural_network, lasagne.layers.Layer):
            self.nn = neural_network
            # find the input layer and get the input variable
            l = self.nn
            while True:
                if isinstance(l, lasagne.layers.InputLayer):
                    self.input = l.input_var
                    break
                l = l.input_layer
        elif isinstance(neural_network, list):
            input_dim = self.state_space.to_vector(self.state_space.get_initial_state()).shape
            nn = lasagne.layers.InputLayer(shape=(1,) + input_dim)
            self.input = nn.input_var
            for i, n in enumerate(neural_network):
                nn = lasagne.layers.DenseLayer(incoming=nn, num_units=n)

            nn = lasagne.layers.DenseLayer(incoming=nn, num_units=self.action_count, nonlinearity=None)
            self.nn = nn
        else:
            raise ValueError("neural network has to be either a list or lasagne.layers.Layer instance.")

        self.target = T.vector('target')
        self.output = lasagne.layers.get_output(self.nn)
        self.forward = theano.function([self.input], self.output)

        self.loss = lasagne.objectives.squared_error(self.output, self.target)
        self.loss = lasagne.objectives.aggregate(self.loss)

        self.params = lasagne.layers.get_all_params(self.nn, trainable=True)
        self.updates = lasagne.updates.sgd(self.loss, self.params, learning_rate=self.learning_params['LEARNING_RATE'])

        self.train_function = theano.function([self.input, self.target], self.loss, updates=self.updates)

    def get_q(self, state, action=None):
        x = self.state_space.to_vector(state).astype(theano.config.floatX)
        x = x[np.newaxis, :]
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
            y = np.ones(self.action_count, dtype=theano.config.floatX) * value
        else:
            a_id = self.action_space.index(action)
            y = self.forward(x)[0]
            y[a_id] = value

        self.train_function(x, y)


class QLearningAgent(Agent):
    def __init__(self, q_function, action_space, discount_factor, greed_eps):
        Agent.__init__(self, action_space=action_space)
        self.discount_factor = discount_factor
        self.greed_eps = greed_eps
        self.episodes_experienced = 0

        # Create a QFunction instance from the given type
        self.q = q_function

    def perceive(self, state, reward, available_actions, reached_goal_state=False, episode_end=False):
        if self.learning_on:
            # perceive
            if reached_goal_state or episode_end:
                self.episodes_experienced += 1
                if reached_goal_state:
                    self.q.update_q(reward, state, None)

            if self.last_action is not None:  # if not on first trial
                estimated_q_sa = self.last_reward + self.discount_factor * (np.max(self.q.get_q(state, None)))
                self.q.update_q(estimated_q_sa, self.last_state, self.last_action)

        # act
        if reached_goal_state or episode_end:
            action = None
        else:
            if not self.learning_on or np.random.rand() > self.greed_eps.get_value(self):
                available_action_ids = [self.action_space.index(a) for a in available_actions]
                # mask unavailable actions
                q_s = self.q.get_q(state, None)
                mask_a = [i not in available_action_ids for i in range(len(q_s))]
                q_ma = np.ma.masked_array(q_s, mask=mask_a)
                action = self.action_space[q_ma.argmax()]
            else:
                a_id = np.random.choice(len(available_actions))
                action = available_actions[a_id]

        self.last_reward = reward
        self.last_state = state
        self.last_action = action
        return action


