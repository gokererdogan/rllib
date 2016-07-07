import numpy as np

from rllib.space import StateSpace, ActionSpace
from rllib.environment import GameEnvironment
from rllib.environment import PLAYER1, PLAYER2
from rllib.agent import Agent
from rllib.q_learning import QLearningAgent
from rllib.policy_gradient import PolicyGradientAgent

EMPTY = -1


class TicTacToeStateSpace(StateSpace):
    def __init__(self):
        StateSpace.__init__(self)
        pass

    def __len__(self):
        return 3**9

    def index(self, state):
        id = 0
        for i, m in enumerate(state):
            id += (m+1)*(3**i)

        return id

    def get_initial_state(self):
        return np.array([EMPTY, EMPTY, EMPTY,
                         EMPTY, EMPTY, EMPTY,
                         EMPTY, EMPTY, EMPTY], dtype=np.int8)

    def get_reward(self, state):
        # return a tuple with rewards for player1 and player2 respectively
        if self._check_win(state, player=PLAYER1):
            return 1.0, -1.0
        elif self._check_win(state, player=PLAYER2):
            return -1.0, 1.0
        else:
            return 0.0, 0.0

    def is_goal_state(self, state):
        if self._check_win(state, player=PLAYER1) or self._check_win(state, player=PLAYER2):
            return True
        if self._check_draw(state):
            return True
        return False

    def _check_win(self, state, player):
        moves = (state == player).reshape((3, 3))
        # vertical
        if np.any(moves.sum(axis=0) == 3):
            return True

        # horizontal
        if np.any(moves.sum(axis=1) == 3):
            return True

        # diagonal
        if np.trace(moves) == 3 or np.trace(np.fliplr(moves)) == 3:
            return True

        return False

    def _check_draw(self, state):
        if np.count_nonzero(np.logical_or(state == PLAYER1, state == PLAYER2)) == 9:
            return True
        return False

    def to_vector(self, state):
        return state

    def to_string(self, state):
        print "==="
        c = ['-', 'x', 'o']
        board_str = "\n".join(["".join([c[i+1] for i in state[(row*3):((row+1)*3)]]) for row in range(3)])
        if self._check_win(state, PLAYER1):
            board_str += "\nGame finished. Winner is PLAYER1"
        elif self._check_win(state, PLAYER2):
            board_str += "\nGame finished. Winner is PLAYER2"
        elif self._check_draw(state):
            board_str += "\nGame finished with DRAW."

        return board_str


class TicTacToeActionSpace(ActionSpace):
    def __init__(self):
        ActionSpace.__init__(self)

    def __len__(self):
        return 9

    def __getitem__(self, i):
        if i < 0 or i > 8:
            raise ValueError("Action id out of bounds.")

        m = np.zeros(9)
        m[i] = 1.0
        return m

    def index(self, action):
        return action.nonzero()[0][0]

    def to_vector(self, action):
        return action


class TicTacToe(GameEnvironment):
    def __init__(self):
        state_space = TicTacToeStateSpace()
        action_space = TicTacToeActionSpace()

        GameEnvironment.__init__(self, game_state_space=state_space,
                                 action_space=action_space)

    def print_game(self):
        print "==="
        c = ['-', 'x', 'o']
        board_str = "\n".join(["".join([c[i] for i in self.current_state[(row*3):((row+1)*3)]]) for row in range(3)])
        print board_str
        if not self.state_space.is_goal_state(self.current_state):
            "Game finished."

    def _current_state_as_first_player(self):
        if self.turn == PLAYER1:
            return self.current_state
        else:
            state = self.current_state.copy()
            state[self.current_state == PLAYER1] = PLAYER2
            state[self.current_state == PLAYER2] = PLAYER1
            return state

    def _advance(self, move):
        new_state = self.current_state.copy()
        if np.count_nonzero(move) != 1:
            raise ValueError("Move should contain only a single 1.")

        m_id = move.nonzero()[0][0]
        if self.current_state[m_id] != EMPTY:
            raise ValueError("Position already played.")

        new_state[m_id] = self.turn
        return new_state

    def get_available_actions(self):
        a_ids = np.nonzero(self.current_state == EMPTY)[0]
        return [self.action_space[a_id] for a_id in a_ids]


class RandomPlayer(Agent):
    def __init__(self):
        Agent.__init__(self, action_space=[])

    def perceive(self, state, reward, available_actions, reached_goal_state=False, episode_end=False):
        if reached_goal_state:
            return None

        a_id = np.random.choice(len(available_actions))
        return available_actions[a_id]


class SearchOneMoveAheadPlayer(Agent):
    def __init__(self, action_space, strength=1.0):
        Agent.__init__(self, action_space=action_space)
        self.strength = strength

    def _find_vertical_moves(self, state):
        board = (state == 0).reshape((3, 3))
        found_moves = []
        # vertical
        if np.any(board.sum(axis=0) == 2):
            cols = np.where(board.sum(axis=0) == 2)[0]
            for col in cols:
                row = np.where(board[:, col] == 0)[0][0]
                found_moves.append((row*3)+col)

        return found_moves

    def _find_horizontal_moves(self, state):
        board = (state == 0).reshape((3, 3))
        found_moves = []
        # horizontal
        if np.any(board.sum(axis=1) == 2):
            rows = np.where(board.sum(axis=1) == 2)[0]
            for row in rows:
                col = np.where(board[row, :] == 0)[0][0]
                found_moves.append((row*3)+col)

        return found_moves

    def _find_diagonal_moves(self, state):
        board = (state == 0).reshape((3, 3))
        found_moves = []
        # diagonal
        if np.trace(board) == 2:
            i = np.where(np.diag(board) == 0)[0][0]
            found_moves.append(4*i)

        if np.trace(np.fliplr(board)) == 2:
            i = np.where(np.diag(np.fliplr(board)) == 0)[0][0]
            found_moves.append((i*3)+(2-i))

        return found_moves

    def perceive(self, state, reward, available_actions, reached_goal_state=False, episode_end=False):
        if reached_goal_state:
            return None

        available_action_ids = [self.action_space.index(a) for a in available_actions]

        if np.random.rand() > self.strength:
            a_id = np.random.choice(len(available_actions))
            selected_move = available_actions[a_id]
        else:
            found_moves = []
            # moves to win
            found_moves.extend(self._find_vertical_moves(state))
            found_moves.extend(self._find_horizontal_moves(state))
            found_moves.extend(self._find_diagonal_moves(state))
            # moves not to lose
            her_state = state.copy()
            her_state[state == 0] = 1
            her_state[state == 1] = 0
            found_moves.extend(self._find_vertical_moves(her_state))
            found_moves.extend(self._find_horizontal_moves(her_state))
            found_moves.extend(self._find_diagonal_moves(her_state))

            # remove non-empty positions
            found_moves = [m for m in found_moves if m in available_action_ids]

            if len(found_moves) > 0:
                m = np.random.choice(found_moves)
                selected_move = np.zeros(9, dtype=np.int8)
                selected_move[m] = 1
            else:
                a_id = np.random.choice(len(available_actions))
                selected_move = available_actions[a_id]

        return selected_move


class HumanPlayer(Agent):
    def __init__(self):
        Agent.__init__(self, action_space=[])

    def perceive(self, state, reward, available_actions, reached_goal_state=False, episode_end=False):
        if reached_goal_state:
            return None
        m = input("Please input your move (0-8): ")
        move = np.zeros(9, dtype=np.int8)
        move[m] = 1
        return move


def pit_against_q_learner(player, epoch_count=20, games_per_epoch=2000):
    q_norm = np.zeros(epoch_count)
    q_dist = np.zeros(epoch_count)
    rewards = np.zeros(epoch_count)
    old_q = q_learner.q.copy()
    for e in range(epoch_count):
        for i in range(games_per_epoch):
            if np.random.rand() > 0.5:
                s, a1, r1, a2, r2 = env.run([q_learner, player], np.inf)
            else:
                s, a2, r2, a1, r1 = env.run([player, q_learner], np.inf)
            rewards[e] += np.sum(r1)
        rewards[e] /= games_per_epoch
        q_dist[e] = np.sum(np.square(q_learner.q - old_q))
        q_norm[e] = np.sum(np.square(q_learner.q))
        old_q = q_learner.q.copy()
        print("Epoch {0:d}| Avg. reward per game: {1:f}, change in Q table: {2:f}, "
              "Q table norm: {3:f}".format(e+1, rewards[e], q_dist[e], q_norm[e]))
    return rewards, q_dist, q_norm


def pit_against_pg_learner(player, epoch_count=20, games_per_epoch=2000):
    wa_norm = np.zeros(epoch_count)
    wa_dist = np.zeros(epoch_count)
    rewards = np.zeros(epoch_count)
    old_wa = pg_learner.wa.get_value().copy()
    for e in range(epoch_count):
        for i in range(games_per_epoch):
            if np.random.rand() > 0.5:
                s, a1, r1, a2, r2 = env.run([pg_learner, player], np.inf)
            else:
                s, a2, r2, a1, r1 = env.run([player, pg_learner], np.inf)
            rewards[e] += np.sum(r1)
        rewards[e] /= games_per_epoch
        wa_dist[e] = np.sum(np.square(pg_learner.wa.get_value() - old_wa))
        wa_norm[e] = np.sum(np.square(pg_learner.wa.get_value()))
        old_wa = pg_learner.wa.get_value().copy()
        print("Epoch {0:d}| Avg. reward per game: {1:f}, change in weights: {2:f}, "
              "weight matrix norm: {3:f}".format(e+1, rewards[e], wa_dist[e], wa_norm[e]))
    return rewards, wa_dist, wa_norm

if __name__ == "__main__":

    env = TicTacToe()

    q_learner = QLearningAgent(env.state_space, env.action_space, discount_factor=1.0,
                               greed_eps=0.25, learning_rate=0.1)

    pg_learner = PolicyGradientAgent(env.state_space, env.action_space, learning_rate=0.005,
                                     update_freq=1, apply_baseline=False, clip_gradients=False,
                                     optimizer='gd')

    ps1 = SearchOneMoveAheadPlayer(env.action_space, strength=0.25)
    ps2 = SearchOneMoveAheadPlayer(env.action_space, strength=0.50)
    ps3 = SearchOneMoveAheadPlayer(env.action_space, strength=0.75)
    ps4 = SearchOneMoveAheadPlayer(env.action_space, strength=1.00)
    pr = RandomPlayer()
    ph = HumanPlayer()

    """
    r, qd, qn = pit_against_q_learner(pr, epoch_count=20, games_per_epoch=2000)

    q_learner.set_learning_mode(False)
    r, _, _ = pit_against_q_learner(pr, epoch_count=1, games_per_epoch=2000)

    print q_learner.q[0]

    s, a1, r1, a2, r2 = env.run([ph, ps4], np.inf, True)
    """

    r, pgd, pgn = pit_against_pg_learner(ps3, epoch_count=20, games_per_epoch=2000)

    pg_learner.set_learning_mode(False)
    r, _, _ = pit_against_pg_learner(ps3, epoch_count=1, games_per_epoch=2000)
