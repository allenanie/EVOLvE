import math
import scipy
import numpy as np
from banditbench.tasks.mab.env import Bandit, MultiArmedBandit
from banditbench.tasks.cb.env import State


class MABAgent:
    name: str

    def __init__(self, env: Bandit) -> None:
        self.k_arms = env.num_arms

    def act(self) -> int:
        """Same as performing a sampling step."""
        raise NotImplementedError

    def update(self, action: int, reward: float) -> None:
        """The action performs an update step based on the action it chose, and the reward it received."""
        raise NotImplementedError


class UCBAgent(MABAgent):
    """alpha-UCB, where alpha is the exploration bonus coefficient"""
    name: str = "UCB"

    def __init__(self, env: MultiArmedBandit, alpha: float = 2.0) -> None:
        super().__init__(env)
        self.actions = list(range(self.k_arms))
        self.alpha = alpha
        self.k_arms = len(self.actions)
        self.reset()

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

    def update(self, action: int, reward: float) -> None:
        self.arms[action] += 1
        self.rewards[action] += reward


class GreedyAgent(UCBAgent):
    """
    This class shows how we can just override the calculate_arm_value() method to implement a different agent.
    """
    name: str = "Greedy"

    def __init__(self, env: MultiArmedBandit) -> None:
        super().__init__(env)

    def calculate_arm_value(self, arm: int) -> float:
        return self.rewards[arm] / self.arms[arm]


class ThompsonSamplingAgent(UCBAgent):
    name: str = "ThompsonSampling"

    def __init__(self, env: MultiArmedBandit, alpha_prior: float = 1.0, beta_prior: float = 1.0) -> None:
        self.alpha_prior = alpha_prior
        self.beta_prior = beta_prior
        super().__init__(env)
        self.reset()

    def reset(self):
        self.alpha = [self.alpha_prior] * self.k_arms
        self.beta = [self.beta_prior] * self.k_arms

    def select_arm(self) -> int:
        samples = [
            scipy.stats.beta.rvs(self.alpha[arm], self.beta[arm])
            for arm in range(self.k_arms)
        ]
        return int(np.argmax(samples))

    def update(self, action: int, reward: float) -> None:
        self.alpha[action] += reward
        self.beta[action] += 1 - reward
