"""
rllib - Reinforcement Learning Library

Policy Gradient. Implementation of policy gradient method for learning policies.
Provides functionality for implementing policies over discrete and continuous action spaces.

Goker Erdogan
https://github.com/gokererdogan
"""

import numpy as np
import scipy.stats
import theano
import theano.tensor as T
import lasagne

from agent import Agent


def _get_input_layer(nn):
    """
    Find the input layer of a lasagne neural network.

    Parameters:
        nn (lasagne.layers.Layer)

    Returns:
        lasagne.layers.InputLayer
    """
    # find the input layer and get the input variable
    l = nn
    input_layer = None
    while True:
        if isinstance(l, lasagne.layers.InputLayer):
            input_layer = l
            break
        l = l.input_layer
    if input_layer is None:
        raise ValueError("Neural network does not have an input layer.")

    return input_layer


class PolicyFunction(object):
    """
    PolicyFunction base class. This abstract class specifies the interface expected of policy functions. A
    policy-function is a function that defines a probability distribution over actions for each state.
    """
    def __init__(self, state_space, action_space):
        """
        Parameters:
            state_space (StateSpace)
            action_space (ActionSpace)
        """
        self.state_space = state_space
        self.action_space = action_space

    def get_action(self, state, available_actions=None):
        """
        Return action for state among available actions.

        Parameters:
            state
            available_actions (list)

        Returns:
            action
        """
        raise NotImplementedError()

    def get_action_probability(self, state, action=None):
        """
        Returns action probabilities for state. This method is used by MHEnvironment
        (environments with Metropolis-Hastings dynamics) to calculate acceptance ratios.

        Parameters:
            state: State for which the action probabilities are requested.
            action: Action for which the probability is requested. If None, probabilities for all actions
            possible in state are returned.

        Returns:
            float or list: A float or list of action probability(ies)
        """
        pass

    def accumulate_reward(self, reward):
        """
        Accumulate reward. This method is called each time the agent is in a new state.

        Parameters:
            reward (float)
        """
        raise NotImplementedError()

    def accumulate_gradients(self, state, action):
        """
        Accumulate gradient. This method is called each time the agent picks an action (when learning is on).

        Parameters:
            state
            action
        """
        raise NotImplementedError()

    def end_episode(self):
        """
        End episode. This method is called once at the end of every episode.
        """
        raise NotImplementedError()

    def update_parameters(self):
        """
        Update parameters of policy function using accumulated gradients
        """
        raise NotImplementedError()


class PolicyNeuralNetwork(PolicyFunction):
    """
    PolicyNeuralNetwork class. This abstract class implements a policy function using a neural network. Since the
    outputs of a policy function depend on the action space (finite vs. infinite) and desired action distribution
    (e.g., multinomial, normal, exponential), the structure of the network (e.g., nonlinearities used in last layer) and
    log probability calculations depend on the choice of action distribution. This class implements the common machinery
    needed for any neural network policy function. This class needs to be subclassed to implement distribution specific
    policy functions (e.g., get_action, accumulate_gradients). See PolicyNeuralNetworkMultinomial for one example.
    """
    def __init__(self, neural_network, state_space, action_space, learning_rate, optimizer):
        """
        Parameters:
            neural_network (lasagne.layers.Layer): A lasagne.layers.Layer instance with input and output layers with
                the proper sizes and ranges, and names input and output respectively.
            state_space (StateSpace)
            action_space (ActionSpace)
            learning_rate (float): Learning rate
            optimizer (lasagne.updates method)
        """
        PolicyFunction.__init__(self, state_space, action_space)
        # create learning rate as a theano share variable because we may want to update it during training
        self.learning_rate = theano.shared(value=np.cast[theano.config.floatX](learning_rate), name='learning_rate')
        self.optimizer = optimizer

        # check neural network input shape
        input_layer = _get_input_layer(neural_network)
        input_dim = state_space.to_vector(state_space.get_initial_state()).shape
        if input_layer.shape != (1,) + input_dim:
            raise ValueError("Neural network input layer does not have expected shape.")

        self.nn = neural_network

        # NOTE ---------
        # Subclasses call this init after they set the following variables
        #   - selected_action: a theano shared variable for keeping track of last selected action
        #   - log_p: a theano function calculating the log probability of an action
        # Hence, these variables are available below.
        # --------------

        # theano code for training
        # theano variables needed for training
        self.input = input_layer.input_var
        self.output = lasagne.layers.get_output(self.nn)
        self.reward = T.scalar('reward')
        # number of collected samples (i.e., episodes) so far. this is used for scaling the gradient estimate
        self.samples_collected = theano.shared(value=np.cast[theano.config.floatX](0.0), name='samples_collected')
        self.params = lasagne.layers.get_all_params(self.nn, trainable=True)
        # shared variables for accumulating gradients
        self.episode_reward = theano.shared(value=np.cast[theano.config.floatX](0.0))  # total reward in an episode
        self.episode_grads = [theano.shared(value=np.zeros_like(p.get_value(), dtype=theano.config.floatX))
                              for p in self.params]
        self.total_grads = [theano.shared(value=np.zeros_like(p.get_value(), dtype=theano.config.floatX))
                            for p in self.params]

        # calculate gradients of log probability. NOTE log_p is set by subclass!
        self.grads = T.grad(self.log_p, self.params)

        # theano functions
        self._forward = theano.function([self.input], self.output)

        # accumulate gradients during an episode
        episode_grad_updates = [(eg, eg + g) for eg, g in zip(self.episode_grads, self.grads)]
        self._update_episode_grads = theano.function([self.input],
                                                     None, updates=episode_grad_updates)

        # accumulate rewards during an episode
        episode_reward_update = [(self.episode_reward, self.episode_reward + self.reward)]
        self._update_episode_reward = theano.function([self.reward], None, updates=episode_reward_update)

        # update total gradients after an episode ends
        end_episode_updates = [(tg, tg + (self.episode_reward * eg)) for tg, eg in
                               zip(self.total_grads, self.episode_grads)]
        end_episode_updates += [(self.samples_collected, self.samples_collected + 1.0)]
        self._update_end_episode = theano.function([], None, updates=end_episode_updates)

        # update neural network parameters
        # Note the negative learning rate (we are maximizing, not minimizing) and that we scale the gradients by
        # samples_collected.
        gradient_updates = self.optimizer(self.total_grads, self.params,
                                          learning_rate=-1.0 * self.learning_rate / self.samples_collected)

        self._update_params = theano.function([], None, updates=gradient_updates)

    def get_action(self, state, available_actions=None):
        """
        Return action for state among available actions.
        Since this operation depends on which distribution we sample actions from, this method needs to be implemented
        by subclasses.

        Parameters:
            state
            available_actions (list)

        Returns:
            action
        """
        raise NotImplementedError()

    def get_action_probability(self, state, action=None):
        """
        Returns action probabilities for state. This method is used by MHEnvironment
        (environments with Metropolis-Hastings dynamics) to calculate acceptance ratios.

        Parameters:
            state: State for which the action probabilities are requested.
            action: Action for which the probability is requested. If None, probabilities for all actions
            possible in state are returned.

        Returns:
            float or list: A float or list of action probability(ies)
        """
        pass

    def accumulate_reward(self, reward):
        self._update_episode_reward(np.cast[theano.config.floatX](reward))

    def accumulate_gradients(self, state, action):
        """
        Accumulate gradient. This method is called each time the agent picks an action (when learning is on).
        Since how gradients are calculated depends on the action distribution, this method is left for subclasses to
        fill in.

        Parameters:
            state
            action
        """
        raise NotImplementedError()

    def end_episode(self):
        """
        End episode. This method is called once at the end of every episode.
        """
        # accumulate episode gradients into total gradients and increment samples collected
        self._update_end_episode()
        # reset episode reward and gradients for next episode
        self.episode_reward.set_value(np.cast[theano.config.floatX](0.0))
        for g in self.episode_grads:
            g.set_value(np.zeros_like(g.get_value(), dtype=theano.config.floatX))

    def update_parameters(self):
        """
        Update parameters of policy function using accumulated gradients
        """
        sample_count = self.samples_collected.get_value()
        if np.isclose(sample_count, 0.0):
            raise RuntimeError("Cannot update using 0 samples.")
        """
        print "params before"
        for p in self.params:
            print p.get_value()
        print "total grads"
        for g in self.total_grads:
            print g.get_value()
        print "samples collected:" + str(sample_count)
        """
        self._update_params()
        # reset total gradients
        for g in self.total_grads:
            g.set_value(np.zeros_like(g.get_value(), dtype=theano.config.floatX))
        # reset the number of samples collected
        self.samples_collected.set_value(np.cast[theano.config.floatX](0.0))
        """
        print "params after"
        for p in self.params:
            print p.get_value()
        print
        """


class PolicyNeuralNetworkMultinomial(PolicyNeuralNetwork):
    """
    PolicyNeuralNetworkMultinomial class. This class implements a policy function that maps a state to a multinomial
    distribution, from which an action is picked. This class is used for implementing policies for finite action
    spaces.
    """
    def __init__(self, neural_network, state_space, action_space, learning_rate, optimizer):
        """
        Parameters:
            neural_network (list or lasagne.layers.Layer): Either a list containing the number of hidden units
                in each hidden layer or a lasagne.layers.Layer instance with input and output layers with the proper
                sizes and names input and output respectively. Output layer needs to have as many units as there are
                actions in action space.
            state_space (StateSpace)
            action_space (ActionSpace)
            learning_rate (float): Learning rate
            optimizer (lasagne.updates method)
        """
        # are we given a lasagne neural network?
        if isinstance(neural_network, lasagne.layers.Layer):
            # check the number of output units
            if neural_network.output_shape[1] != len(action_space):
                raise ValueError("Neural network should have one output unit for each action.")
        elif isinstance(neural_network, list):  # are we given a list of hidden unit counts
            neural_network = self._create_neural_network(neural_network, state_space, action_space)
        else:
            raise ValueError("neural network has to be either a list or lasagne.layers.Layer instance.")

        # theano variables needed specifically for a multinomial action distribution
        self.selected_action = theano.shared(0, 'selected_action')  # the index of the selected action
        # log probability of selected action
        output = lasagne.layers.get_output(neural_network)
        self.log_p = T.log(output[0, self.selected_action])

        # NOTE --------
        # call super's init. NOTE that we call it at the end because super's init needs the variables we set above.
        # -------------
        PolicyNeuralNetwork.__init__(self, neural_network, state_space, action_space,
                                     learning_rate=learning_rate, optimizer=optimizer)

    @staticmethod
    def _create_neural_network(unit_counts, state_space, action_space):
        """
        Create a neural network from hidden unit counts. This method is called to create a neural network when the class
        is initialized with a list of hidden unit counts.
        """
        input_dim = state_space.to_vector(state_space.get_initial_state()).shape
        nn = lasagne.layers.InputLayer(shape=(1,) + input_dim)
        for i, n in enumerate(unit_counts):
            nn = lasagne.layers.DenseLayer(incoming=nn, num_units=n)

        nn = lasagne.layers.DenseLayer(incoming=nn, num_units=len(action_space),
                                       nonlinearity=lasagne.nonlinearities.softmax)
        return nn

    def get_action(self, state, available_actions=None):
        """
        Return action for state among available actions.

        Parameters:
            state
            available_actions (list)

        Returns:
            action
        """
        x = self.state_space.to_vector(state).astype(theano.config.floatX)
        x = x[np.newaxis, :]
        available_action_ids = [self.action_space.index(a) for a in available_actions]
        probs = self._forward(x)[0]
        available_probs = probs[available_action_ids] / np.sum(probs[available_action_ids])
        a_id = np.random.choice(available_action_ids, p=available_probs)
        return self.action_space[a_id]

    def get_action_probability(self, state, action=None):
        """
        Returns action probabilities for state. This method is used by MHEnvironment
        (environments with Metropolis-Hastings dynamics) to calculate acceptance ratios.

        Parameters:
            state: State for which the action probabilities are requested.
            action: Action for which the probability is requested. If None, probabilities for all actions
            possible in state are returned.

        Returns:
            float or list: A float or list of action probability(ies)
        """
        x = self.state_space.to_vector(state).astype(theano.config.floatX)
        x = x[np.newaxis, :]
        probs = self._forward(x)[0]
        if action is None:
            return probs
        else:
            return probs[self.action_space.index(action)]

    def accumulate_gradients(self, state, action):
        """
        Accumulate gradient. This method is called each time the agent picks an action (when learning is on).

        Parameters:
            state
            action
        """
        x = self.state_space.to_vector(state).astype(theano.config.floatX)
        x = x[np.newaxis, :]
        a_id = self.action_space.index(action)
        self.selected_action.set_value(a_id)
        self._update_episode_grads(x)


class PolicyNeuralNetworkNormal(PolicyNeuralNetwork):
    """
    PolicyNeuralNetworkNormal class. This class implements a policy function that maps a state to a (diagonal) normal
    distribution, from which an action is picked. This class is used for implementing policies for real-valued action
    spaces.
    """
    def __init__(self, neural_network, state_space, action_space, learning_rate, optimizer, cov_type="identity",
                 std_dev=1.0):
        """
        Parameters:
            neural_network (list or lasagne.layers.Layer): Either a list containing the number of hidden units
                in each hidden layer or a lasagne.layers.Layer instance with input and output layers with the proper
                sizes and names input and output respectively. Output of the network should have
                    - If cov_type is diagonal: 2*(no of dimensions of action space) output units, where the first half
                        is used as the mean and the rest as std. deviation for the normal distribution.
                    - If cov_type is identity: (no of dimensions of action space) output units where each unit specifies
                        the mean for one dimension.
            state_space (StateSpace)
            action_space (ActionSpace)
            learning_rate (float): Learning rate
            optimizer (lasagne.updates method)
            cov_type (string): identity or diagonal.
            std_dev (float): Standard deviation for identity covariance matrix, i.e., sigma = std_dev*I
        """
        if cov_type not in ['identity', 'diagonal']:
            raise ValueError("Covariance type should be identity or diagonal.")
        if cov_type == 'identity':
            self.std_dev = std_dev
        self.cov_type = cov_type

        self.action_dim = int(np.prod(action_space.shape()))

        # are we given a lasagne neural network?
        if isinstance(neural_network, lasagne.layers.Layer):
            # check the number of output units. we need one set of outputs for mean and another set for std. deviations.
            if cov_type == 'diagonal' and neural_network.output_shape[1] != 2 * self.action_dim:
                raise ValueError("Neural network should have one mean output unit and one variance output unit for "
                                 "each action dimension.")
            elif cov_type == 'identity' and neural_network.output_shape[1] != self.action_dim:
                raise ValueError("Neural network should have one mean output unit for "
                                 "each action dimension.")
        elif isinstance(neural_network, list):  # are we given a list of hidden unit counts
            neural_network = self._create_neural_network(neural_network, state_space, action_space, cov_type)
        else:
            raise ValueError("neural network has to be either a list or lasagne.layers.Layer instance.")

        # theano variables needed specifically for a normal action distribution
        self.selected_action = theano.shared(np.zeros(self.action_dim), 'selected_action')
        # log probability of selected action
        output = lasagne.layers.get_output(neural_network)
        output_mean = output[0, 0:self.action_dim]
        if self.cov_type == 'diagonal':
            output_log_std = output[0, self.action_dim:]
            output_std = T.exp(output_log_std)

            self.log_p = (-0.5 * T.sum(2.0 * output_log_std)) + \
                         T.sum(-0.5 * T.square((self.selected_action - output_mean) / output_std))

            # initialize the weights for variance units to 0.
            w = neural_network.W.get_value()
            w[:, self.action_dim:] = 0.0
            neural_network.W.set_value(w)
        else:  # identity
            self.log_p = T.sum(-0.5 * T.square((self.selected_action - output_mean) / self.std_dev))

        # NOTE --------
        # call super's init. NOTE that we call it at the end because super's init needs the variables we set above.
        # -------------
        PolicyNeuralNetwork.__init__(self, neural_network, state_space, action_space,
                                     learning_rate=learning_rate, optimizer=optimizer)

    @staticmethod
    def _create_neural_network(unit_counts, state_space, action_space, cov_type):
        """
        Create a neural network from hidden unit counts. This method is called to create a neural network when the class
        is initialized with a list of hidden unit counts.
        """
        input_dim = state_space.to_vector(state_space.get_initial_state()).shape
        action_dim = int(np.prod(action_space.shape()))
        if cov_type == 'identity':
            output_dim = action_dim
        elif cov_type == 'diagonal':
            output_dim = 2*action_dim
        else:
            raise ValueError("Covariance type should be identity or diagonal.")

        nn = lasagne.layers.InputLayer(shape=(1,) + input_dim)
        for i, n in enumerate(unit_counts):
            nn = lasagne.layers.DenseLayer(incoming=nn, num_units=n, W=lasagne.init.Normal(0.01))

        nn = lasagne.layers.DenseLayer(incoming=nn, num_units=output_dim, W=lasagne.init.Normal(0.01),
                                       nonlinearity=lasagne.nonlinearities.linear)
        return nn

    def get_action(self, state, available_actions=None):
        """
        Return action for state by sampling from a normal distribution. Mean and variance of this distribution
        is calculated using the neural network.
        Note that available actions are ignored.

        Parameters:
            state
            available_actions (list)

        Returns:
            action
        """
        x = self.state_space.to_vector(state).astype(theano.config.floatX)
        x = x[np.newaxis, :]
        output = self._forward(x)[0]
        mean = output[0:self.action_dim]

        if self.cov_type == 'diagonal':
            std = np.exp(output[self.action_dim:])
        else:
            std = self.std_dev

        action = mean + std * np.random.randn(self.action_dim)
        return action

    def get_action_probability(self, state, action=None):
        """
        Returns action probabilities for state. This method is used by MHEnvironment
        (environments with Metropolis-Hastings dynamics) to calculate acceptance ratios.

        Parameters:
            state: State for which the action probabilities are requested.
            action: Action for which the probability is requested. Note that action cannot be None because we cannot
            return the probabilities for all actions (which are infinitely many)

        Returns:
            float: Action probability
        """
        if action is None:
            raise ValueError("Action cannot be None for a Gaussian action distribution.")
        x = self.state_space.to_vector(state).astype(theano.config.floatX)
        x = x[np.newaxis, :]
        output = self._forward(x)[0]
        mean = output[0:self.action_dim]
        if self.cov_type == 'diagonal':
            std = np.exp(output[self.action_dim:])
        else:
            std = self.std_dev

        return scipy.stats.multivariate_normal.pdf(action, mean, np.square(std))

    def accumulate_gradients(self, state, action):
        """
        Accumulate gradient. This method is called each time the agent picks an action (when learning is on).

        Parameters:
            state
            action
        """
        x = self.state_space.to_vector(state).astype(theano.config.floatX)
        x = x[np.newaxis, :]
        self.selected_action.set_value(np.array(action).astype(theano.config.floatX))
        self._update_episode_grads(x)


class PolicyGradientAgent(Agent):
    """
    PolicyGradientAgent class. This class implements an agent that uses policy gradient method to learn policies.
    """
    def __init__(self, policy_function, discount_factor, update_freq=1):
        """
        Parameters:
            policy_function (PolicyFunction): Policy function instance implementing the mapping from states to actions.
            discount_factor (float): Reward discount factor
            update_freq (int): Update frequency. Parameters are updated after every update_freq episodes
        """
        Agent.__init__(self, action_space=policy_function.action_space)

        if discount_factor < 0.0 or discount_factor > 1.0:
            raise ValueError("Discount factor should be between 0.0 and 1.0.")
        self.discount_factor = discount_factor

        if update_freq <= 0:
            raise ValueError("Update frequency should be positive.")
        self.update_freq = update_freq

        self.policy_function = policy_function
        self.trials_experienced = 0  # used for discounting reward
        self.episodes_experienced = 0  # used for updating parameters at the desired update_freq

    def reset(self):
        pass

    def get_action(self, state, available_actions=None):
        action = self.policy_function.get_action(state, available_actions)
        return action

    def get_action_probability(self, state, action=None):
        probs = self.policy_function.get_action_probability(state, action)
        return probs

    def perceive(self, state, reward, available_actions, reached_goal_state=False, episode_end=False):
        """
        Perceive and act. Uses policy gradient to update parameters of policy function.

        Parameters:
            state
            reward (float)
            available_actions (list)
            reached_goal_state (bool)
            episode_end (bool)

        Returns:
            action
        """
        if self.learning_on:
            # perceive
            self.trials_experienced += 1
            self.policy_function.accumulate_reward(self.discount_factor**(self.trials_experienced-1) * reward)

            if reached_goal_state or episode_end:
                self.trials_experienced = 0
                self.episodes_experienced += 1
                self.policy_function.end_episode()

                if self.episodes_experienced % self.update_freq == 0:
                    self.policy_function.update_parameters()

        # do not act if it is a terminal state
        if reached_goal_state or episode_end:
            return None

        # act
        action = self.get_action(state, available_actions)

        if self.learning_on:
            # accumulate gradients
            self.policy_function.accumulate_gradients(state, action)

        return action
