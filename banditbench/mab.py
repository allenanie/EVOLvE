from pydantic import BaseModel
from typing import Dict, Any, Tuple, Union, List, Optional
import numpy as np
from banditbench.tasks.mab_scenarios import BanditScenario, BanditConfig

State = Union[None, Any]
BernArmParam = float
GaussianArmParam = Tuple[float, float]
BanditArmParam = Union[BernArmParam, GaussianArmParam]

Action = Union[int, str]
ExpectedReward = float


class Interaction(BaseModel):
    action: Action
    expected_reward: ExpectedReward
    is_random: Union[bool, None] = None

    def __init__(self, action: Action, expected_reward: ExpectedReward, is_random: Union[bool, None] = None) -> None:
        super().__init__(action=action, expected_reward=expected_reward, is_random=is_random)


class Bandit:
    num_arms: int
    horizon: int
    seed: Optional[int]
    h: int

    def reset(self) -> Any:
        raise NotImplementedError

    def step(self, action: int) -> Tuple[State, float, bool, Dict[str, Any]]:
        raise NotImplementedError

    def reward_fn(self, context: State, action: int) -> float:
        raise NotImplementedError

    def set_seed(self, seed: Optional[int] = None) -> None:
        self.seed = seed
        self.np_random = np.random.default_rng(self.seed)


class MultiArmedBandit(Bandit):
    arm_params: List[BanditArmParam]
    history: List[Interaction]
    instance_hardness: float  # this is the Delta_min or "gap" in the paper

    def __init__(self, num_arms: int, horizon: int, arm_params: List[BanditArmParam], seed: Optional[int] = None,
                 instance_hardness: float = 0.0) -> None:
        self.num_arms = num_arms
        self.horizon = horizon
        self.arm_params = arm_params
        self.set_seed(seed)
        self.instance_hardness = instance_hardness
        self.validate()
        if seed is not None:
            self.shuffle_arms()
        self.initialize_defaults()


    def initialize_defaults(self) -> None:
        """Initialize default values for type-annotated attributes, but are instance variables"""
        self.history = []
        self.h = 0

    def shuffle_arms(self) -> None:
        self.np_random.shuffle(self.arm_params)

    def validate(self) -> None:
        assert len(
            self.arm_params) == self.num_arms, f"Number of arm parameters {len(self.arm_params)} does not match number of arms {self.num_arms}."
        assert self.horizon > 0, f"Horizon {self.horizon} must be positive."

    def step(self, action: int) -> Tuple[State, float, bool, Dict[str, Any]]:
        assert action >= 0 and action < self.num_arms, f"Action {action} is not in the action space [{0}, {self.num_arms}]."
        if self.h >= self.horizon:
            raise ValueError(
                "Episode is done. Call env.reset() to start a new episode."
            )

        info = {'arm_param': self.arm_params[action]}

        self.h += 1
        done = self.h == self.horizon
        reward = self.reward_fn(None, action)
        self.history.append(Interaction(action, self.expected_reward(action)))

        return None, reward, done, info

    def reset(self):
        self.history = []
        self.h = 0


class BernoulliBandit(MultiArmedBandit):

    def __init__(
            self,
            num_arms: int,
            horizon: int,
            arm_params: List[BernArmParam],
            seed: Optional[int] = None,  # might remove this
    ):
        # instance_hardness is computed by getting the highest arm and subtracting the second highest arm
        instance_hardness = max(arm_params) - sorted(arm_params)[-2]
        super(BernoulliBandit, self).__init__(num_arms=num_arms, horizon=horizon, arm_params=arm_params, seed=seed,
                                              instance_hardness=instance_hardness)

    def reward_fn(self, context: State, action: int) -> float:
        return 1 if self.np_random.uniform(0, 1) < self.arm_params[action] else 0

    def expected_reward(self, action: int) -> float:
        return self.arm_params[action]


def compute_simple_kl(p_mu, q_mu, sigma):
    return (sigma ** 2 + (p_mu - q_mu) ** 2) / (2 * sigma ** 2) - 1 / 2


def compute_general_kl(p_mu, p_sigma, q_mu, q_sigma):
    return np.log(q_sigma / p_sigma) + (p_sigma ** 2 + (p_mu - q_mu) ** 2) / (2 * q_sigma ** 2) - 0.5


class GaussianBandit(MultiArmedBandit):

    def __init__(self, num_arms: int, horizon: int, arm_params: List[GaussianArmParam],
                 seed: Optional[int] = None) -> None:
        # to make difficulty easy to assess, we require all arms to have the same variance
        assert all(p[1] == arm_params[0][1] for p in arm_params), "All arms must have the same variance"
        # then we compute the instance hardness as the KL divergence between the best and second best arm
        instance_hardness = compute_simple_kl(max(arm_params)[0], sorted(arm_params)[-2][0], max(arm_params)[1])
        super(GaussianBandit, self).__init__(num_arms=num_arms, horizon=horizon, arm_params=arm_params, seed=seed,
                                             instance_hardness=instance_hardness)

    def reward_fn(self, context: State, action: int) -> float:
        return self.np_random.normal(self.arm_params[action][0], self.arm_params[action][1])

    def expected_reward(self, action: int) -> float:
        return self.arm_params[action][0]


# Now, we define the LLM-Bandit class, VerbalBandit.
VerbalState = Union[None, str]


class VerbalBandit(Bandit):
    history: List[Interaction]

    def __init__(self,
                 core_bandit: MultiArmedBandit,
                 bandit_scenario: BanditScenario,
                 # ===== arguments for bandit_scenario_cls =====
                 scenario_seed: Optional[int] = None,
                 instruction_type: str = "base",
                 num_fewshot: int = 0, few_shot_config: Optional[BanditConfig] = None) -> None:
        """
        bandit_scenario: Can be one of three types of inputs: a string, a instantiated BanditScenario class, or the class constructor itself
        """
        self.num_arms = core_bandit.num_arms
        self.horizon = core_bandit.horizon
        self.core_bandit = core_bandit
        self.initialize_defaults()

        self.instruction_type = instruction_type

        if isinstance(bandit_scenario, str):
            assert bandit_scenario in ["ButtonPushing", "OnlineAds", "VideoWatching",
                                       "ClothesShopping"], "Unknown bandit scenario"
            self.bandit_scenario = eval(bandit_scenario)(num_actions=self.num_arms,
                                                        num_fewshot=num_fewshot,
                                                        few_shot_config=few_shot_config,
                                                        seed=scenario_seed)
        elif isinstance(bandit_scenario, type):
            # noinspection PyCallingNonCallable
            self.bandit_scenario = bandit_scenario(num_actions=self.num_arms,
                                                   num_fewshot=num_fewshot,
                                                   few_shot_config=few_shot_config,
                                                   seed=scenario_seed)
        elif isinstance(bandit_scenario, BanditScenario):
            self.bandit_scenario = bandit_scenario
        else:
            raise ValueError("Unknown bandit scenario")

    def initialize_defaults(self) -> None:
        self.history = []

    def reset(self) -> Any:
        self.core_bandit.reset()

    @property
    def action_names(self) -> List[str]:
        return self.bandit_scenario.action_names

    def step(self, action: str) -> Tuple[VerbalState, float, bool, Dict[str, Any]]:
        """
        action: the action selected by the agent, a string of the actual action name

        For MAB's VerbalState, we return instruction/problem instance, it's fixed for all steps
        """
        assert type(action) == str, "Action must be a string for VerbalBandit"
        # Find matching action name, accounting for case and whitespace
        action = action.strip().lower()
        action_names = [name.strip().lower() for name in self.bandit_scenario.action_names]
        try:
            action_index = action_names.index(action)
            is_random = False
        except ValueError:
            is_random = True
            action_index = int(self.core_bandit.np_random.integers(0, len(self.bandit_scenario.action_names)))

        # now get the reward
        state, reward, done, info = self.core_bandit.step(action_index)
        assert state is None, "State should be None for MultiArmedBandit"

        verbal_instruction = self.bandit_scenario.get_instruction(self.instruction_type)
        self.history.append(Interaction(action, self.core_bandit.expected_reward(action_index), is_random))

        return verbal_instruction, reward, done, {'is_random': is_random}
