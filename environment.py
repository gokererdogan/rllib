import numpy as np

PLAYER1 = 0
PLAYER2 = 1


class Environment(object):
    def __init__(self, state_space, action_space):
        self.state_space = state_space
        self.state_count = len(state_space)

        self.action_space = action_space
        self.action_count = len(action_space)

        self.current_state = None
        self.current_reward = None

    def reset(self):
        self.current_state = self.state_space.get_initial_state()
        self.current_reward = self.state_space.get_reward(self.current_state)

    def get_next_states(self, state, action):
        pass

    def get_available_actions(self):
        return self.action_space

    def _advance(self, action):
        raise NotImplementedError()

    def run(self, agent, episode_length, verbose=False):
        self.reset()
        agent.reset()

        # we adopt the convention in Section 21 of Artificial Intelligence by Russell, Norvig
        # s_0, s_1, ..., s_T
        states = []
        # a_t is the action taken at state s_t
        # a_0, a_1, ..., a_T-1, a_T=None
        actions = []
        # r_t is the reward received associated with state s_t
        # r_0, r_2, ..., r_T
        rewards = []

        if verbose:
            print self.state_space.to_string(self.current_state)

        e = 0
        while True:
            action = agent.perceive(self.current_state, self.current_reward, self.get_available_actions(),
                                    reached_goal_state=self.state_space.is_goal_state(self.current_state),
                                    episode_end=False)

            states.append(self.current_state)
            actions.append(action)
            rewards.append(self.current_reward)
            self.current_state = self._advance(action)
            self.current_reward = self.state_space.get_reward(self.current_state)

            if verbose:
                print self.state_space.to_string(self.current_state)

            e += 1
            if e >= episode_length or self.state_space.is_goal_state(self.current_state):
                # let the agent perceive one last time.
                # note that we append None as its action because the agent does not act in the terminal state
                _ = agent.perceive(self.current_state, self.current_reward, self.get_available_actions(),
                                   reached_goal_state=self.state_space.is_goal_state(self.current_state),
                                   episode_end=(e >= episode_length))
                states.append(self.current_state)
                rewards.append(self.current_reward)
                actions.append(None)
                break

        return np.array(states), np.array(actions), np.array(rewards)


class GameEnvironment(Environment):
    def __init__(self, game_state_space, action_space):
        Environment.__init__(self, state_space=game_state_space, action_space=action_space)

        self.turn = None

    def reset(self):
        Environment.reset(self)
        # we want the current_reward to be mutable
        self.current_reward = list(self.current_reward)
        self.turn = PLAYER1

    def _advance(self, action):
        raise NotImplementedError()

    def _current_state_as_first_player(self):
        # because players do not know if they are the first or second player,
        # they need to see the current state as if they are the first player
        # this method does the necessary conversion
        raise NotImplementedError()

    def run(self, agents, episode_length, verbose=False):
        self.reset()
        agents[PLAYER1].reset()
        agents[PLAYER2].reset()

        # one states list for both players
        states = []
        # one action and reward list for each player
        actions = [[], []]
        rewards = [[], []]

        if verbose:
            print self.state_space.to_string(self.current_state)

        e = 0
        while True:
            action = agents[self.turn].perceive(self._current_state_as_first_player(),
                                                self.current_reward[self.turn],
                                                self.get_available_actions(),
                                                reached_goal_state=self.state_space.is_goal_state(self.current_state),
                                                episode_end=False)

            states.append(self.current_state)
            actions[self.turn].append(action)
            rewards[self.turn].append(self.current_reward[self.turn])

            # play player's move
            self.current_state = self._advance(action)
            # rewards is a list with 2 elements (reward for player 1 and 2 respectively)
            new_rewards = self.state_space.get_reward(self.current_state)
            # zero current player's reward. she already knows about her reward
            self.current_reward[self.turn] = new_rewards[self.turn]
            # now it is next player's turn
            self.turn = (self.turn + 1) % 2
            # accumulate the reward for the next player because she needs to know the reward for the move
            # other player played
            self.current_reward[self.turn] += new_rewards[self.turn]

            if verbose:
                print self.state_space.to_string(self.current_state)

            e += 1
            if e >= episode_length or self.state_space.is_goal_state(self.current_state):
                states.append(self.current_state)
                # let both players (who does not know the game ended) perceive one last time.
                for p in [PLAYER1, PLAYER2]:
                    self.turn = p
                    _ = agents[p].perceive(self._current_state_as_first_player(),
                                           self.current_reward[p],
                                           self.get_available_actions(),
                                           reached_goal_state=self.state_space.is_goal_state(self.current_state),
                                           episode_end=(e >= episode_length))
                    rewards[p].append(self.current_reward[p])
                    actions[p].append(None)

                break

        return np.array(states), np.array(actions[PLAYER1]), np.array(rewards[PLAYER1]), \
               np.array(actions[PLAYER2]), np.array(rewards[PLAYER2])

