import cPickle as pkl

import numpy as np
import theano
import theano.ifelse
import theano.tensor as T

from lasagne.updates import adam, rmsprop

PLAYER1 = 1
PLAYER2 = 2

LOSE = -1
DRAW = 0
WIN = 1
PLAYING = None


class TicTacToe(object):
    def __init__(self):
        self.board = np.zeros(3*3, dtype=np.int8)
        self.current_turn = PLAYER1
        self.is_playing = True
        self.winner = None

    def reset(self):
        self.board[:] = 0
        self.current_turn = PLAYER1
        self.is_playing = True

    def print_game(self):
        print "==="
        c = ['-', 'x', 'o']
        board_str = "\n".join(["".join([c[i] for i in self.board[(row*3):((row+1)*3)]]) for row in range(3)])
        print board_str
        if not self.is_playing:
            if self.winner is None:
                print "Game finished with DRAW."
            else:
                print "Game finished. Winner is PLAYER {0:d}".format(self.winner)

    def play_move(self, player, move):
        if np.count_nonzero(move) != 1:
            raise ValueError("Move should contain only a single 1.")

        if self.current_turn != player:
            raise ValueError("It is not player {0:d}'s turn.".format(player))

        if not self.is_playing:
            raise ValueError("Game is not started. Please call reset.")

        if self.board[move.nonzero()] != 0:
            raise ValueError("Position already played.")

        self.board[move.nonzero()] = player
        self.current_turn = (self.current_turn % 2) + 1

        if self.check_win(player):
            self.winner = player
            self.is_playing = False
            return WIN

        if self.check_draw():
            self.winner = None
            self.is_playing = False
            return DRAW

        return PLAYING

    def check_win(self, player):
        moves = (self.board == player).reshape((3, 3))
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

    def check_draw(self):
        if np.count_nonzero(self.board) == 9:
            return True
        return False

    def get_empty_positions(self):
        if not self.is_playing:
            raise ValueError("Game is not started. Please call reset.")

        return np.nonzero(self.board == 0)[0]


class Player(object):
    def __init__(self, player_id):
        self.player_id = player_id

    def set_player_id(self, pid):
        self.player_id = pid

    def get_move(self, game):
        raise NotImplementedError()

    def end_game(self, result):
        raise NotImplementedError()


class NeuralNetworkPlayer(Player):
    def __init__(self, player_id=PLAYER1, hidden_unit_count=10, learning_rate=0.001, update_freq=1, optimizer='gd'):
        Player.__init__(self, player_id=player_id)

        self.player_id = player_id
        self.hidden_unit_count = hidden_unit_count
        self.learning_rate = theano.shared(value=learning_rate, name='learning_rate')
        self.update_freq = theano.shared(value=update_freq, name='update_freq')
        self.optimizer = optimizer
        self.games_played = theano.shared(value=0)
        self.moves_played = theano.shared(value=0)
        self.total_reward = theano.shared(value=0.0)

        self.whns = []
        self.bhns = []
        self.wans = []
        self.bans = []
        self.selected_probs = []

        self.state = T.vector('state')
        self.reward = T.scalar('reward')
        self.selected_action = T.iscalar('selected_action')
        self.wh = theano.shared(value=0.2*np.random.uniform(-1.0, 1.0, (9, hidden_unit_count)), name='wh')
        self.bh = theano.shared(value=0.2*np.random.uniform(-1.0, 1.0, hidden_unit_count), name='bh')
        self.wa = theano.shared(value=0.2*np.random.uniform(-1.0, 1.0, (hidden_unit_count, 9)), name='wa')
        self.ba = theano.shared(value=0.2*np.random.uniform(-1.0, 1.0, 9), name='wa')
        self.params = [self.wh, self.bh, self.wa, self.ba]

        self.h = T.tanh(T.dot(self.state, self.wh) + self.bh),
        self.action = T.nnet.softmax(T.dot(self.h, self.wa) + self.ba)
        self.logp = T.log(self.action[0, self.selected_action])
        self.forward = theano.function([self.state], self.action)

        # ----- TEMP
        self.get_hidden = theano.function([self.state], self.h)

        self.game_dwh = theano.shared(value=np.zeros((9, hidden_unit_count)))
        self.game_dbh = theano.shared(value=np.zeros(hidden_unit_count))
        self.game_dwa = theano.shared(value=np.zeros((hidden_unit_count, 9)))
        self.game_dba = theano.shared(value=np.zeros(9))

        # these contain the negatives of the total gradients
        self.total_dwh = theano.shared(value=np.zeros((9, hidden_unit_count)))
        self.total_dbh = theano.shared(value=np.zeros(hidden_unit_count))
        self.total_dwa = theano.shared(value=np.zeros((hidden_unit_count, 9)))
        self.total_dba = theano.shared(value=np.zeros(9))
        self.grads = [self.total_dwh, self.total_dbh, self.total_dwa, self.total_dba]

        self.dwh, self.dbh, self.dwa, self.dba = T.grad(self.logp, [self.wh, self.bh, self.wa, self.ba])

        get_move_updates = [(self.game_dwh, self.game_dwh + self.dwh), (self.game_dbh, self.game_dbh + self.dbh),
                            (self.game_dwa, self.game_dwa + self.dwa), (self.game_dba, self.game_dba + self.dba),
                            (self.moves_played, self.moves_played + 1)]
        self.update_get_move = theano.function([self.state, self.selected_action], None, updates=get_move_updates)

        end_game_updates = [(self.games_played, self.games_played + 1),
                            (self.total_reward, self.total_reward + self.reward),
                            (self.total_dwh, self.total_dwh + (-self.game_dwh * self.reward / self.moves_played)),
                            (self.total_dbh, self.total_dbh + (-self.game_dbh * self.reward / self.moves_played)),
                            (self.total_dwa, self.total_dwa + (-self.game_dwa * self.reward / self.moves_played)),
                            (self.total_dba, self.total_dba + (-self.game_dba * self.reward / self.moves_played))]
        self.update_end_game = theano.function([self.reward], None, updates=end_game_updates)

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

        self.gradient_step = theano.function([], None, updates=gradient_updates)

    def reset(self):
        self.games_played.set_value(0)
        self.total_reward.set_value(0.0)

    def zero_total_gradients(self):
        self.total_dwh.set_value(np.zeros((9, self.hidden_unit_count)))
        self.total_dbh.set_value(np.zeros(self.hidden_unit_count))
        self.total_dwa.set_value(np.zeros((self.hidden_unit_count, 9)))
        self.total_dba.set_value(np.zeros(9))

    def zero_game_gradients(self):
        self.game_dwh.set_value(np.zeros((9, self.hidden_unit_count)))
        self.game_dbh.set_value(np.zeros(self.hidden_unit_count))
        self.game_dwa.set_value(np.zeros((self.hidden_unit_count, 9)))
        self.game_dba.set_value(np.zeros(9))
        self.moves_played.set_value(0)

    def _get_action_probs(self, state):
        return self.forward(state).ravel()

    def get_move(self, game):
        state = game.board.astype(dtype=theano.config.floatX)
        state[state != 0] -= 1.5
        if self.player_id == PLAYER1:
            state *= -1.0
        probs = self._get_action_probs(state)
        available_moves = game.get_empty_positions()
        m = np.zeros(9, dtype=np.int8)
        p = probs[available_moves]
        mid = np.random.choice(available_moves, p=(p / p.sum()))
        m[mid] = 1

        self.selected_probs.append(probs[mid])

        # accumulate gradients
        self.update_get_move(state, mid)

        return m

    def end_game(self, result):
        """
        Parameters:
            result (int): -1 LOSE, 0 DRAW, 1 WIN
        """

        self.selected_probs.append(-1.0)

        self.update_end_game(result)
        self.zero_game_gradients()

        if int(self.games_played.get_value()) % int(self.update_freq.get_value()) == 0:
            # --- temp
            whn = np.sum(np.square(self.total_dwh.get_value()))
            bhn = np.sum(np.square(self.total_dbh.get_value()))
            wan = np.sum(np.square(self.total_dwa.get_value()))
            ban = np.sum(np.square(self.total_dba.get_value()))
            self.whns.append(whn)
            self.bhns.append(bhn)
            self.wans.append(wan)
            self.bans.append(ban)

            self.gradient_step()
            self.zero_total_gradients()

    def save(self, filename):
        pkl.dump(self, open(filename, 'w'))

    @staticmethod
    def load(filename):
        pkl.load(open(filename, 'rb'))

    def __getstate__(self):
        weights = [p.get_value() for p in self.params]
        return weights + [self.learning_rate, self.update_freq]

    def __setstate(self, state):
        wh_val, bh_val, wa_val, ba_val, lr, uf = state
        self.wh.set_value(wh_val)
        self.bh.set_value(bh_val)
        self.wa.set_value(wa_val)
        self.ba.set_value(ba_val)
        self.learning_rate = lr
        self.update_freq = uf


class RandomPlayer(Player):
    def __init__(self, player_id=PLAYER1):
        Player.__init__(self, player_id)

    def get_move(self, game):
        m = np.zeros(9, dtype=np.int8)
        m[np.random.choice(game.get_empty_positions())] = 1
        return m

    def end_game(self, result):
        pass


class SearchOneMoveAheadPlayer(Player):
    def __init__(self, player_id=PLAYER1, strength=1.0):
        Player.__init__(self, player_id)
        self.strength = strength

    def _find_vertical_moves(self, board):
        found_moves = []
        # vertical
        if np.any(board.sum(axis=0) == 2):
            cols = np.where(board.sum(axis=0) == 2)[0]
            for col in cols:
                row = np.where(board[:, col] == 0)[0][0]
                found_moves.append((row*3)+col)

        return found_moves

    def _find_horizontal_moves(self, board):
        found_moves = []
        # horizontal
        if np.any(board.sum(axis=1) == 2):
            rows = np.where(board.sum(axis=1) == 2)[0]
            for row in rows:
                col = np.where(board[row, :] == 0)[0][0]
                found_moves.append((row*3)+col)

        return found_moves

    def _find_diagonal_moves(self, board):
        found_moves = []
        # diagonal
        if np.trace(board) == 2:
            i = np.where(np.diag(board) == 0)[0][0]
            found_moves.append(4*i)

        if np.trace(np.fliplr(board)) == 2:
            i = np.where(np.diag(np.fliplr(board)) == 0)[0][0]
            found_moves.append((i*3)+(2-i))

        return found_moves

    def get_move(self, game):
        selected_move = np.zeros(9, dtype=np.int8)
        available_moves = game.get_empty_positions()

        if np.random.rand() > self.strength:
            selected_move[np.random.choice(available_moves)] = 1
        else:
            found_moves = []
            # moves to win
            my_board = (game.board == self.player_id).reshape((3, 3))
            found_moves.extend(self._find_vertical_moves(my_board))
            found_moves.extend(self._find_horizontal_moves(my_board))
            found_moves.extend(self._find_diagonal_moves(my_board))
            # moves not to lose
            her_board = (game.board == (self.player_id % 2) + 1).reshape((3, 3))
            found_moves.extend(self._find_vertical_moves(her_board))
            found_moves.extend(self._find_horizontal_moves(her_board))
            found_moves.extend(self._find_diagonal_moves(her_board))

            # remove non-empty positions
            found_moves = [m for m in found_moves if m in available_moves]

            if len(found_moves) > 0:
                m = np.random.choice(found_moves)
                selected_move[m] = 1
            else:
                selected_move[np.random.choice(available_moves)] = 1

        return selected_move

    def end_game(self, result):
        pass


class HumanPlayer(Player):
    def __init__(self, player_id=PLAYER1):
        Player.__init__(self, player_id)

    def get_move(self, game):
        m = input("Please input your move (0-8): ")
        move = np.zeros(9, dtype=np.int8)
        move[m] = 1
        return move

    def end_game(self, result):
        pass


def play_game(game, player1, player2, verbose=False):
    players = [player1, player2]
    player1.set_player_id(PLAYER1)
    player2.set_player_id(PLAYER2)
    game_state = game.play_move(game.current_turn, players[game.current_turn-1].get_move(game))
    if verbose:
        game.print_game()
    while game_state == PLAYING:
        game_state = game.play_move(game.current_turn, players[game.current_turn-1].get_move(game))
        if verbose:
            game.print_game()

    player1_result = DRAW
    player2_result = DRAW
    if game.winner == 1:
        player1_result = WIN
        player2_result = LOSE
    elif game.winner == 2:
        player1_result = LOSE
        player2_result = WIN

    player1.end_game(player1_result)
    player2.end_game(player2_result)

    return player1_result, player2_result


def pit_against(player1, player2, epoch_count=5, games_per_epoch=5000):
    p1_results = np.zeros(epoch_count)
    p2_results = np.zeros(epoch_count)
    g = TicTacToe()
    for e in range(epoch_count):
        p1_epoch_avg = 0.0
        p2_epoch_avg = 0.0
        for i in range(games_per_epoch):
            if np.random.rand() > .5:
                p1r, p2r = play_game(g, player1, player2)
            else:
                p2r, p1r = play_game(g, player2, player1)
            p1_epoch_avg += p1r
            p2_epoch_avg += p2r

            g.reset()

        p1_results[e] = p1_epoch_avg / games_per_epoch
        p2_results[e] = p2_epoch_avg / games_per_epoch
        print("|Epoch {0:d}| Player 1 average: {1:f}, Player 2 average: {2:f}".format(e+1, p1_results[e], p2_results[e]))

    return p1_results, p2_results


if __name__ == "__main__":
    g = TicTacToe()
    ps1 = SearchOneMoveAheadPlayer(strength=0.25)
    ps2 = SearchOneMoveAheadPlayer(strength=0.50)
    ps3 = SearchOneMoveAheadPlayer(strength=0.75)
    ps4 = SearchOneMoveAheadPlayer(strength=1.00)
    pr = RandomPlayer()

    pnn = NeuralNetworkPlayer(learning_rate=0.01, hidden_unit_count=50, update_freq=50, optimizer='adam')

    pit_against(pr, pnn, epoch_count=5)
    """
    pit_against(ps1, pnn, epoch_count=5)
    pit_against(ps2, pnn, epoch_count=5)
    pit_against(ps3, pnn, epoch_count=5)
    pit_against(ps4, pnn, epoch_count=5)
    """

