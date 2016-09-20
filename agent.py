"""
rllib - Reinforcement Learning Library

Agent class

Goker Erdogan
https://github.com/gokererdogan
"""


class Agent(object):
    """
    Base Agent class.
    """
    def __init__(self, action_space, learning_on=True):
        """
        Parameters:
            action_space (ActionSpace): Action space for agent. Since different agents can have different actions
                available to them, we make action space an attribute of Agent.
        """
        self.action_space = action_space

        self.learning_on = learning_on

    def reset(self):
        """
        Reset agent. This method is called by :class:`rllib.environment.Environment` before each episode.
        """
        raise NotImplementedError()

    def set_learning_mode(self, learning_on):
        self.learning_on = learning_on

    def get_action_probabilities(self, state):
        """
        Returns action probabilities for state. This method is used by dynamic programming methods such as
        :func:`rllib.rl.evaluate_policy_dp`.

        Parameters:
            state: State for which the action probabilities are requested.

        Returns:
            list: A list of action probabilities
        """
        pass

    def perceive(self, state, reward, available_actions, reached_goal_state=False, episode_end=False):
        """
        Perceive state and act. This method should be overridden in child classes and needs to return the action taken
        by the agent.

        Parameters:
            state: Current state
            reward (float): Reward for current state
            available_actions (list): List of actions that can be taken in current state.
            reached_goal_state (bool)
            episode_end (bool)

        Returns:
            action
        """
        raise NotImplementedError()

