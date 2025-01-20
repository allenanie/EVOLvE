import math
import scipy
import numpy as np
from banditbench.tasks.mab.env import MultiArmedBandit
from banditbench.tasks.cb.env import ContextualBandit
from banditbench.tasks.cb.env import State

from typing import Union, Dict, Any

from banditbench.agents.types import MABAgent, CBAgent
from banditbench.sampling.sampler import SampleBase


class UCBAgent(MABAgent, SampleBase):
    """alpha-UCB, where alpha is the exploration bonus coefficient"""
    name: str = "UCB"

    def __init__(self, env: MultiArmedBandit, alpha: float = 0.5) -> None:
        super().__init__(env)
        self.actions = list(range(self.k_arms))
        self.alpha = alpha

        self.arms = [0] * self.k_arms  # Number of times each arm has been pulled
        self.rewards = [0] * self.k_arms  # Accumulated rewards for each arm
        self.exploration_bonuses = [0] * self.k_arms
        self.exploitation_values = [0] * self.k_arms

    def reset(self):
        self.arms = [0] * self.k_arms  # Number of times each arm has been pulled
        self.rewards = [0] * self.k_arms  # Accumulated rewards for each arm
        self.exploration_bonuses = [0] * self.k_arms
        self.exploitation_values = [0] * self.k_arms

    def calculate_arm_value(self, arm: int) -> float:
        exploration_bonus = self.calculate_exp_bonus(arm)
        exploitation_value = self.calculate_exp_value(arm)

        self.exploration_bonuses[arm] = exploration_bonus
        self.exploitation_values[arm] = exploitation_value

        return exploitation_value + exploration_bonus

    def calculate_exp_bonus(self, arm):
        # return math.sqrt(1.0 / self.arms[arm])
        return math.sqrt((self.alpha * math.log(sum(self.arms))) / self.arms[arm])

    def calculate_exp_value(self, arm):
        return self.rewards[arm] / self.arms[arm]

    def select_arm(self) -> int:
        """
        Select an arm to pull. Note that we only use self.calculate_arm_value() to select the arm.
        """
        # a hard exploration rule to always explore an arm at least once
        for arm in range(self.k_arms):
            if self.arms[arm] == 0:
                return arm  # Return an unexplored arm

        # if all arms have been explored, use UCB to select the arm
        arm_values = [self.calculate_arm_value(arm) for arm in range(self.k_arms)]
        return int(np.argmax(arm_values))

    def act(self) -> int:
        return self.select_arm()

    def update(self, action: int, reward: float, info: Dict[str, Any]) -> None:
        self.arms[action] += 1
        self.rewards[action] += reward


class GreedyAgent(UCBAgent, SampleBase):
    """
    This class shows how we can just override the calculate_arm_value() method to implement a different agent.
    """
    name: str = "Greedy"

    def __init__(self, env: MultiArmedBandit) -> None:
        super().__init__(env)

    def calculate_exp_value(self, arm: int) -> float:
        return self.rewards[arm] / self.arms[arm]


class ThompsonSamplingAgent(UCBAgent, SampleBase):
    name: str = "ThompsonSampling"

    def __init__(self, env: MultiArmedBandit, alpha_prior: float = 1.0, beta_prior: float = 1.0) -> None:
        self.alpha_prior = alpha_prior
        self.beta_prior = beta_prior
        super().__init__(env)
        self.alpha = [self.alpha_prior] * self.k_arms
        self.beta = [self.beta_prior] * self.k_arms

    def reset(self):
        self.alpha = [self.alpha_prior] * self.k_arms
        self.beta = [self.beta_prior] * self.k_arms

    def select_arm(self) -> int:
        samples = [
            scipy.stats.beta.rvs(self.alpha[arm], self.beta[arm])
            for arm in range(self.k_arms)
        ]
        return int(np.argmax(samples))

    def update(self, action: int, reward: float, info: Dict[str, Any]) -> None:
        self.alpha[action] += reward
        self.beta[action] += 1 - reward


# ======= Contextual Bandit ==========

class LinUCBAgent(CBAgent, SampleBase):
    name: str = 'LinUCB'

    def __init__(self, env: ContextualBandit, alpha: float = 0.5):
        super().__init__(env)

        self.d = env.feature_dim
        self.alpha = alpha
        self.A = [np.identity(self.d) for _ in range(self.k_arms)]
        self.b = [np.zeros((self.d, 1)) for _ in range(self.k_arms)]

    def reset(self):
        # init must be called before reset
        self.A = [np.identity(self.d) for _ in range(self.k_arms)]
        self.b = [np.zeros((self.d, 1)) for _ in range(self.k_arms)]

    def act(self, state: State) -> int:
        """Same as performing a sampling step."""
        action = self.select_action(state)
        return action

    def select_action(self, state: State) -> int:
        context = state.feature
        context = context.reshape(-1, 1)
        ucb_values = []

        ucb_exploitation_values = []
        ucb_exploration_bonuses = []

        for a in range(self.k_arms):
            A_inv = np.linalg.inv(self.A[a])
            theta = A_inv.dot(self.b[a])

            ucb = theta.T.dot(context) + self.alpha * np.sqrt(
                context.T.dot(A_inv).dot(context)
            )
            ucb_values.append(ucb[0, 0])

            ucb_exploitation_values.append(theta.T.dot(context)[0, 0])
            exp_v = self.alpha * np.sqrt(context.T.dot(A_inv).dot(context))
            ucb_exploration_bonuses.append(exp_v[0, 0])

        # tie-breaking arbitrarily
        candidate_arms = []
        highest_ucb = -1
        for arm_index in range(self.k_arms):
            # If current arm is highest than current highest_ucb
            arm_ucb = ucb_values[arm_index]
            # because we have the precision of 0.01
            if arm_ucb - highest_ucb > 0.001:
                # Set new max ucb
                highest_ucb = arm_ucb
                # Reset candidate_arms list with new entry based on current arm
                candidate_arms = [arm_index]
            # If there is a tie, append to candidate_arms
            if highest_ucb - arm_ucb <= 1e-5:
                candidate_arms.append(arm_index)

        # Choose based on candidate_arms randomly (tie breaker)
        chosen_arm_index = np.random.choice(candidate_arms)

        return int(chosen_arm_index)

    def update(self, state: State, action: int, reward: float, info: Dict[str, Any]) -> None:
        context = state.feature
        context = context.reshape(-1, 1)
        self.A[action] += context.dot(context.T)
        self.b[action] += reward * context

# add an AgentBuilder
# and a classicAgent here
# for delayed execution
