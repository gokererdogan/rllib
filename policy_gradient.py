import cPickle as pkl

import numpy as np
import theano
import theano.ifelse
import theano.tensor as T

from lasagne.updates import adam, rmsprop

from agent import Agent


class PolicyGradientAgent(Agent):
    def __init__(self, state_space, action_space, learning_rate=0.001, update_freq=1, optimizer='gd'):
        Agent.__init__(self, action_space=action_space)

        self.state_space = state_space
        self.input_dim = len(self.state_space)
        self.output_dim = self.action_count

        self.learning_rate = theano.shared(value=learning_rate, name='learning_rate')
        self.update_freq = theano.shared(value=update_freq, name='update_freq')
        self.optimizer = optimizer
        self.episodes_experienced = theano.shared(value=0)
        self.actions_executed = theano.shared(value=0)  # in a single episode
        self.episode_reward = theano.shared(value=0.0)  # total reward in an episode
        self.total_actions_executed = theano.shared(value=0)
        self.total_reward = theano.shared(value=0.0)

        self.state = T.vector('state')
        self.reward = T.scalar('reward')
        self.selected_action = T.iscalar('selected_action')
        self.wa = theano.shared(value=0.2*np.random.uniform(-1.0, 1.0, (self.input_dim, self.output_dim)), name='wa')
        self.ba = theano.shared(value=0.2*np.random.uniform(-1.0, 1.0, self.output_dim), name='ba')
        self.params = [self.wa, self.ba]

        self.action = T.nnet.softmax(T.dot(self.state, self.wa) + self.ba)
        self.logp = T.log(self.action[0, self.selected_action])
        self.forward = theano.function([self.state], self.action)

        # these contain the total gradient for current episode
        self.episode_dwa = theano.shared(value=np.zeros((self.input_dim, self.output_dim)))
        self.episode_dba = theano.shared(value=np.zeros(self.output_dim))

        # these contain the negatives of the total gradients
        self.total_dwa = theano.shared(value=np.zeros((self.input_dim, self.output_dim)))
        self.total_dba = theano.shared(value=np.zeros(self.output_dim))
        self.grads = [self.total_dwa, self.total_dba]

        self.dwa, self.dba = T.grad(self.logp, [self.wa, self.ba])

        # accumulate gradients during an episode
        episode_grad_updates = [(self.episode_dwa, self.episode_dwa + self.dwa),
                                (self.episode_dba, self.episode_dba + self.dba),
                                (self.actions_executed, self.actions_executed + 1),
                                (self.total_actions_executed, self.total_actions_executed + 1)]
        self.update_episode_grads = theano.function([self.state, self.selected_action],
                                                    None, updates=episode_grad_updates)

        # accumulate rewards during an episode
        episode_reward_update = [(self.episode_reward, self.episode_reward + self.reward),
                                 (self.total_reward, self.total_reward + self.reward)]
        self.update_episode_reward = theano.function([self.reward], None, updates=episode_reward_update)

        # update total gradients after an episode ends
        end_episode_updates = [(self.episodes_experienced, self.episodes_experienced + 1),
                               (self.total_dwa,
                                self.total_dwa + (-self.episode_dwa * self.episode_reward / self.actions_executed)),
                               (self.total_dba,
                                self.total_dba + (-self.episode_dba * self.episode_reward / self.actions_executed))]
        self.update_end_episode = theano.function([], None, updates=end_episode_updates)

        # update neural network parameters
        if self.optimizer == 'adam':
            gradient_updates = adam([g / self.update_freq for g in self.grads], self.params,
                                    learning_rate=self.learning_rate)
        elif self.optimizer == 'rmsprop':
            gradient_updates = rmsprop([g / self.update_freq for g in self.grads], self.params,
                                       learning_rate=self.learning_rate)
        elif self.optimizer == 'gd':
            gradient_updates = [(p, p - self.learning_rate * g / self.update_freq)
                                for p, g in zip(self.params, self.grads)]
        else:
            raise ValueError("Optimizer can be gd, rmsprop, or adam.")

        self.update_params = theano.function([], None, updates=gradient_updates)

    def zero_after_update_params(self):
        self.total_dwa.set_value(np.zeros((self.input_dim, self.output_dim)))
        self.total_dba.set_value(np.zeros(self.output_dim))

    def zero_after_episode_end(self):
        self.episode_dwa.set_value(np.zeros((self.input_dim, self.output_dim)))
        self.episode_dba.set_value(np.zeros(self.output_dim))
        self.episode_reward.set_value(0.0)
        self.actions_executed.set_value(0)

    def _get_action_probs(self, state):
        return self.forward(state).ravel()

    def _state_to_input(self, state):
        s_id = self.state_space.index(state)
        x = np.zeros(self.input_dim, dtype=theano.config.floatX)
        x[s_id] = 1.0
        return x

    def perceive(self, state, reward, reached_goal_state=False, episode_end=False):
        x = self._state_to_input(state)

        # perceive
        self.update_episode_reward(reward)

        if reached_goal_state or episode_end:
            self.update_end_episode()
            self.zero_after_episode_end()

            if int(self.episodes_experienced.get_value()) % int(self.update_freq.get_value()) == 0:
                self.update_params()
                self.zero_after_update_params()

        # act
        probs = self._get_action_probs(x)
        a_id = np.random.choice(self.action_count, p=probs)

        # accumulate gradients
        self.update_episode_grads(x, a_id)

        return self.action_space[a_id]

    def save(self, filename):
        pkl.dump(self, open(filename, 'w'))

    @staticmethod
    def load(filename):
        pkl.load(open(filename, 'rb'))

    def __getstate__(self):
        weights = [p.get_value() for p in self.params]
        return weights + [self.learning_rate, self.update_freq]

    def __setstate(self, state):
        vals, ba_val, lr, uf = state
        for p, val in zip(self.params, vals):
            p.set_value(val)
        self.learning_rate = lr
        self.update_freq = uf


