import math
import scipy
import numpy as np
from pydantic import BaseModel
from typing import List, Dict, Any, Union
from banditbench.mab import State, Bandit, MultiArmedBandit


class Agent:
    name: str

    def __init__(self, env: Bandit) -> None:
        self.k_arms = env.num_arms

    def act(self, obs: State) -> int:
        """Same as performing a sampling step."""
        raise NotImplementedError

    def update(self, obs: State, action: int, reward: float) -> None:
        """The action performs an update step based on the action it chose, and the reward it received."""
        raise NotImplementedError

    def get_guide_info(self) -> Dict[str, Any]:
        raise NotImplementedError("This agent does not provide any guide info.")


class ActionInfoField(BaseModel):
    info_name: str  # such as "exploitation value"
    info_template: Union[str, None] # Note, we only support non-key-value templates, like "value name = {:.2f}"
    value: Union[float, str]

    def __init__(self, info_name: str, value: Union[float, str], info_template: Union[str, None] = None):
        super().__init__(info_name=info_name, value=value, info_template=info_template)

    def __str__(self):
        if self.info_template is None:
            if isinstance(self.value, float):
                return f"{self.info_name} {self.value:.2f}"
            else:
                return f"{self.info_name} {self.value}"
        else:
            return self.info_template.format(self.value)

    def to_str(self):
        return str(self)
    
    def __add__(self, other: Union['ActionInfoField', 'ActionInfo']):
        if isinstance(other, ActionInfoField):
            return ActionInfo(action_info_fields=[self, other])
        elif isinstance(other, ActionInfo):
            return ActionInfo(action_info_fields=self.action_infos + other.action_infos)
        else:
            raise ValueError(f"Unsupported type: {type(other)}")

class ActionInfo(BaseModel):
    # an action can have multiple fields (of information)
    action_info_fields: List[ActionInfoField]

    def __str__(self):
        return ", ".join([info.to_str() for info in self.action_info_fields])

    def to_str(self):
        return str(self)
    
    def __len__(self):
        return len(self.action_infos)
    
    def __add__(self, other: Union['ActionInfo', 'ActionInfoField']):
        if isinstance(other, ActionInfoField):
            return ActionInfo(action_info_fields=self.action_info_fields + [other])
        elif isinstance(other, ActionInfo):
            return ActionInfo(action_info_fields=self.action_info_fields + other.action_info_fields)
        else:
            raise ValueError(f"Unsupported type: {type(other)}")

class UCBAgent(Agent):
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

    def act(self, obs: State) -> int:
        return self.select_arm()

    def update(self, obs: State, action: int, reward: float) -> None:
        self.arms[action] += 1
        self.rewards[action] += reward

    def get_guide_info(self) -> List[ActionInfo]:
        actions_info = []
        for arm in range(self.k_arms):
            exploration_bonus = self.calculate_exp_bonus(arm) if self.arms[arm] > 0 else "inf"
            exp_bonus_guide = ActionInfoField(info_name='exploration_bonus', value=exploration_bonus)

            exploitation_value = self.calculate_exp_value(arm) if self.arms[arm] > 0 else 0
            exp_value_guide = ActionInfoField(info_name='exploitation_value', value=exploitation_value)

            actions_info.append(exp_bonus_guide + exp_value_guide)

        assert len(actions_info) == len(self.actions)
        return actions_info


class GreedyAgent(UCBAgent):
    """
    This class shows how we can just override the calculate_arm_value() method to implement a different agent.
    """
    name: str = "Greedy"

    def __init__(self, env: MultiArmedBandit) -> None:
        super().__init__(env)

    def calculate_arm_value(self, arm: int) -> float:
        return self.rewards[arm] / self.arms[arm]

    def get_guide_info(self) -> List[ActionInfo]:
        actions_info = []
        for arm in range(self.k_arms):
            exploitation_value = self.calculate_exp_value(arm) if self.arms[arm] > 0 else 0
            exp_value_guide = ActionInfoField(info_name='exploitation_value', value=exploitation_value)
            actions_info.append(exp_value_guide)
        return actions_info

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

    def update(self, obs: State, action: int, reward: float) -> None:
        self.alpha[action] += reward
        self.beta[action] += 1 - reward

    def get_guide_info(self) -> List[ActionInfo]:
        actions_info = []
        for arm in range(self.k_arms):
            alpha = self.alpha[arm]
            beta = self.beta[arm]
            p = scipy.stats.beta.rvs(self.alpha[arm], self.beta[arm])
            alpha_guide = ActionInfoField(info_name='alpha', value=alpha, info_template='Prior Beta Distribution(alpha={:.2f}')
            beta_guide = ActionInfoField(info_name='beta', value=beta, info_template='beta={:.2f})')
            probability_guide = ActionInfoField(info_name='probability', value=p, info_template='Posterior Bernoulli p={:.2f}')

            actions_info.append(alpha_guide + beta_guide + probability_guide)

        assert len(actions_info) == len(self.actions)
        return actions_info