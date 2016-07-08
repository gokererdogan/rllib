import cPickle as pkl

import numpy as np
import theano
import theano.ifelse
import theano.tensor as T

from lasagne.updates import adam, rmsprop

from agent import Agent


def get_initial_parameter_value(shape):
    return 0.2*np.random.uniform(-1.0, 1.0, shape).astype(dtype=theano.config.floatX)


class PolicyGradientAgent(Agent):
    def __init__(self, state_space, action_space, greed_eps, learning_rate=0.001, update_freq=1, apply_baseline=False,
                 clip_gradients=False, optimizer='gd'):
        Agent.__init__(self, action_space=action_space)

        self.state_space = state_space
        # we assume the state is represented by a real vector of fixed length
        self.input_dim = self.state_space.to_vector(self.state_space.get_initial_state()).size
        self.output_dim = self.action_count

        self.greed_eps = greed_eps
        self.learning_rate = theano.shared(value=np.cast[theano.config.floatX](learning_rate), name='learning_rate')
        self.update_freq = update_freq
        self.apply_baseline = apply_baseline
        self.clip_gradients = clip_gradients
        self.optimizer = optimizer
        self.episodes_experienced_t = theano.shared(value=np.cast[theano.config.floatX](0.0))
        self.episodes_experienced = 0  # we have two variables, one for theano, one for external use
        self.actions_executed = theano.shared(value=np.cast[theano.config.floatX](0.0))  # in a single episode
        self.episode_reward = theano.shared(value=np.cast[theano.config.floatX](0.0))  # total reward in an episode
        self.total_actions_executed = theano.shared(value=np.cast[theano.config.floatX](0.0))
        self.total_reward = theano.shared(value=np.cast[theano.config.floatX](0.0))
        self.baseline = theano.shared(value=np.cast[theano.config.floatX](0.0))

        self.state = T.vector('state')
        self.reward = T.scalar('reward')
        self.selected_action = T.iscalar('selected_action')
        self.wa = theano.shared(value=get_initial_parameter_value((self.input_dim, self.output_dim)), name='wa')
        self.ba = theano.shared(value=get_initial_parameter_value(self.output_dim), name='ba')
        self.params = [self.wa, self.ba]

        self.action = T.nnet.softmax(T.dot(self.state, self.wa) + self.ba)
        self.logp = T.log(self.action[0, self.selected_action])
        self.forward = theano.function([self.state], self.action)

        # these contain the total gradient for current episode
        self.episode_dwa = theano.shared(value=np.zeros((self.input_dim, self.output_dim), dtype=theano.config.floatX))
        self.episode_dba = theano.shared(value=np.zeros(self.output_dim, dtype=theano.config.floatX))

        # these contain the negatives of the total gradients
        self.total_dwa = theano.shared(value=np.zeros((self.input_dim, self.output_dim), dtype=theano.config.floatX))
        self.total_dba = theano.shared(value=np.zeros(self.output_dim, dtype=theano.config.floatX))
        self.grads = [self.total_dwa, self.total_dba]

        # store gradient magnitudes (mainly for debugging)
        self.grad_magnitudes = [[] for _ in self.grads]

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

        # update baseline
        baseline_update = [(self.baseline, self.total_reward / self.total_actions_executed)]
        self.update_baseline = theano.function([], None, updates=baseline_update)

        # update total gradients after an episode ends
        end_episode_updates = [(self.episodes_experienced_t, self.episodes_experienced_t + 1),
                               (self.total_dwa,
                                self.total_dwa +
                                (-self.episode_dwa *
                                 ((self.episode_reward / self.actions_executed) - self.baseline))),
                               (self.total_dba,
                                self.total_dba +
                                (-self.episode_dba *
                                 ((self.episode_reward / self.actions_executed) - self.baseline)))]
        self.update_end_episode = theano.function([], None, updates=end_episode_updates)

        # clip gradients
        clip_gradient_updates = [(g, theano.ifelse.ifelse(T.sum(T.square(g)) > 1.0, g / T.sum(T.square(g)), g))
                                 for g in self.grads]
        self.update_clip_gradients = theano.function([], None, updates=clip_gradient_updates)

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

    def reset(self):
        # reset is called by Environment at the start of each episode
        self.zero_after_episode_end()
        self.zero_after_update_params()

    def zero_after_update_params(self):
        self.total_dwa.set_value(np.zeros((self.input_dim, self.output_dim), dtype=theano.config.floatX))
        self.total_dba.set_value(np.zeros(self.output_dim, dtype=theano.config.floatX))

    def zero_after_episode_end(self):
        self.episode_dwa.set_value(np.zeros((self.input_dim, self.output_dim), dtype=theano.config.floatX))
        self.episode_dba.set_value(np.zeros(self.output_dim, dtype=theano.config.floatX))
        self.episode_reward.set_value(np.cast[theano.config.floatX](0.0))
        self.actions_executed.set_value(np.cast[theano.config.floatX](0.0))

    def _record_grad_magnitudes(self):
        for i, g in enumerate(self.grads):
            self.grad_magnitudes[i].append(np.sum(np.square(g.get_value())))

    def _get_action_probs(self, state):
        return self.forward(np.cast[theano.config.floatX](state)).ravel()

    def perceive(self, state, reward, available_actions, reached_goal_state=False, episode_end=False):
        x = self.state_space.to_vector(state)

        if self.learning_on:
            # perceive
            self.update_episode_reward(np.cast[theano.config.floatX](reward))
            if self.apply_baseline:
                self.update_baseline()

            if reached_goal_state or episode_end:
                self.episodes_experienced += 1
                self.update_end_episode()
                self.zero_after_episode_end()

                if int(self.episodes_experienced_t.get_value()) % self.update_freq == 0:
                    if self.clip_gradients:
                        self.update_clip_gradients()

                    self.update_params()
                    self._record_grad_magnitudes()
                    self.zero_after_update_params()

        # do not act if it is a terminal state
        if reached_goal_state or episode_end:
            return None

        # act
        available_action_ids = [self.action_space.index(a) for a in available_actions]
        if not self.learning_on or np.random.rand() > self.greed_eps.get_value(self):
            probs = self._get_action_probs(x)
            available_probs = probs[available_action_ids] / np.sum(probs[available_action_ids])
            a_id = np.random.choice(available_action_ids, p=available_probs)
        else:
            a_id = np.random.choice(available_action_ids)

        if self.learning_on:
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


