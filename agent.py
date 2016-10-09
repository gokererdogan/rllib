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

    def get_action(self, state, available_actions=None):
        """
        Return an action for given state. This method is intended to be used in perceive method to get
        action for current state.

        Parameters:
            state
            available_actions (list): A list of available actions in state

        Returns:
            action
        """
        raise NotImplementedError()

    def get_action_probability(self, state, action=None):
        """
        Returns action probabilities for state. This method is used by dynamic programming methods such as
        :func:`rllib.rl.evaluate_policy_dp` and by MHEnvironment (environments with Metropolis-Hastings
         dynamics).

        Parameters:
            state: State for which the action probabilities are requested.
            action: Action for which the probability is requested. If None, probabilities for all actions
            possible in state are returned.

        Returns:
            float or list: A float or list of action probability(ies)
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

