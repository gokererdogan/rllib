import numpy as np

from rllib.space import DiscreteStateSpace, DiscreteActionSpace
from rllib.environment import GameEnvironment
from rllib.environment import PLAYER1, PLAYER2
from rllib.agent import Agent
from rllib.parameter_schedule import GreedyEpsilonConstantSchedule
from rllib.q_learning import QLearningAgent, QTableLookup, QNeuralNetwork
from rllib.policy_gradient import PolicyGradientAgent


class SimpleGameStateSpace(DiscreteStateSpace):
    def __init__(self):
        DiscreteStateSpace.__init__(self, states=[0, 1, 2, 3])

    def index(self, state):
        return state

    def get_initial_state(self):
        return 0

    def get_reward(self, state):
        if state == 2:
            return 1.0, -1.0
        elif state == 3:
            return -1.0, 1.0
        else:
            return 0.0, 0.0

    def is_goal_state(self, state):
        return state == 2 or state == 3

    def to_string(self, state):
        return str(state)

    def to_vector(self, state):
        x = np.zeros(4)
        x[state] = 1.0

        return x


class SimpleGameActionSpace(DiscreteActionSpace):
    def __init__(self):
        DiscreteActionSpace.__init__(self, actions=['L', 'R'])

    def to_vector(self, action):
        a = np.zeros(2)
        a[self.index(action)] = 1.0
        return a


class SimpleGame(GameEnvironment):
    def __init__(self):
        state_space = SimpleGameStateSpace()
        action_space = SimpleGameActionSpace()

        GameEnvironment.__init__(self, game_state_space=state_space,
                                 action_space=action_space)

    def _current_state_as_first_player(self):
        if self.turn == PLAYER1:
            return self.current_state
        else:
            if self.current_state in (0, 1):
                return self.current_state
            else:
                if self.current_state == 2:
                    return 3
                else:
                    return 2

    def _advance(self, move):
        if self.current_state == 0:
            if move == 'L':
                return 0
            elif move == 'R':
                return 1
        elif self.current_state == 1:
            if move == 'L':
                return 0
            elif move == 'R':
                if self.turn == PLAYER1:
                    return 2
                elif self.turn == PLAYER2:
                    return 3
        else:
            return self.current_state

    def get_available_actions(self):
        return self.action_space.actions


class RandomPlayer(Agent):
    def __init__(self):
        Agent.__init__(self, action_space=SimpleGameActionSpace())

    def perceive(self, state, reward, available_actions, reached_goal_state=False, episode_end=False):
        if reached_goal_state:
            return None

        return np.random.choice(self.action_space.actions)


class HumanPlayer(Agent):
    def __init__(self):
        Agent.__init__(self, action_space=[])

    def perceive(self, state, reward, available_actions, reached_goal_state=False, episode_end=False):
        if reached_goal_state:
            return None
        m = raw_input("Please input your move (L-R): ")
        return m


def pit_against_q_learner(players, epoch_count=20, games_per_epoch=2000):
    rewards = np.zeros(epoch_count)
    for e in range(epoch_count):
        for i in range(games_per_epoch):
            player = np.random.choice(players)
            if np.random.rand() > 0.5:
                s, a1, r1, a2, r2 = env.run([q_learner, player], np.inf)
            else:
                s, a2, r2, a1, r1 = env.run([player, q_learner], np.inf)
            rewards[e] += (np.sum(r1) / (len(a1) - 1))
        rewards[e] /= games_per_epoch
        print("Epoch {0:d}| Avg. reward per game: {1:f}".format(e+1, rewards[e]))
    return rewards


def pit_against_pg_learner(players, epoch_count=20, games_per_epoch=2000):
    wa_norm = np.zeros(epoch_count)
    wa_dist = np.zeros(epoch_count)
    rewards = np.zeros(epoch_count)
    old_wa = pg_learner.wa.get_value().copy()
    for e in range(epoch_count):
        for i in range(games_per_epoch):
            player = np.random.choice(players)
            if np.random.rand() > 0.5:
                s, a1, r1, a2, r2 = env.run([pg_learner, player], np.inf, False)
            else:
                s, a2, r2, a1, r1 = env.run([player, pg_learner], np.inf, False)
            rewards[e] += (np.sum(r1) / (len(a1) - 1))
        rewards[e] /= games_per_epoch
        wa_dist[e] = np.sum(np.square(pg_learner.wa.get_value() - old_wa))
        wa_norm[e] = np.sum(np.square(pg_learner.wa.get_value()))
        old_wa = pg_learner.wa.get_value().copy()
        print("Epoch {0:d}| Avg. reward per game: {1:f}, change in weights: {2:f}, "
              "weight matrix norm: {3:f}".format(e+1, rewards[e], wa_dist[e], wa_norm[e]))
    return rewards, wa_dist, wa_norm

if __name__ == "__main__":

    env = SimpleGame()

    epoch_count = 10
    games_per_epoch = 1000

    pr = RandomPlayer()
    ph = HumanPlayer()

    eps_schedule = GreedyEpsilonConstantSchedule(eps=0.2)

    # q_function = QTableLookup(env.state_space, env.action_space, learning_rate=0.1)
    q_function = QNeuralNetwork([], env.state_space, env.action_space, learning_params={'LEARNING_RATE': 0.001})
    q_learner = QLearningAgent(q_function, env.action_space, discount_factor=1.0,
                               greed_eps=eps_schedule)

    r = pit_against_q_learner([pr], epoch_count=epoch_count, games_per_epoch=games_per_epoch)

    print q_function.get_q(env.state_space[0])
    print q_function.get_q(env.state_space[1])
    print q_function.get_q(env.state_space[2])
    print q_function.get_q(env.state_space[3])

    """
    pg_learner = PolicyGradientAgent(env.state_space, env.action_space, learning_rate=0.1, greed_eps=eps_schedule,
                                     update_freq=100, apply_baseline=False, clip_gradients=False,
                                     optimizer='gd')

    r, pgd, pgn = pit_against_pg_learner([pr], epoch_count=epoch_count,
                                         games_per_epoch=games_per_epoch)
    """

