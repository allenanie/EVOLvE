from pydantic import BaseModel
from typing import Dict, Any, Tuple, Union, List, Optional

from banditbench.tasks.env import Action, ExpectedReward, Bandit

Info = Union[Dict[str, Any], None]


class State(BaseModel):
    feature: Any  # must be numpy array
    index: Union[int, None]  # a pointer to the dataset (if there is a dataset)
    info: Info = None  # additional information


class Interaction(BaseModel):
    state: State
    action: Action
    expected_reward: ExpectedReward
    is_random: Union[bool, None] = None

    def __init__(self, state: State, action: Action, expected_reward: ExpectedReward,
                 is_random: Union[bool, None] = None) -> None:
        super().__init__(state=state, action=action, expected_reward=expected_reward, is_random=is_random)


class ContextualBandit(Bandit):
    history: List[Interaction]

    def sample_state(self) -> State:
        """
        We sample a state from the state distribution
        """
        raise NotImplementedError

    def reward_fn(self, state: State, action: Action) -> float:
        """In a contextual bandit, this is a function f(x, a)"""
        raise NotImplementedError

    def reset(self) -> Tuple[State, Info]:
        raise NotImplementedError

    def step(self, state: State, action: Action) -> Tuple[State, float, bool, Dict[str, Any]]:
        raise NotImplementedError

    @property
    def verbal_info(self) -> Dict[str, Any]:
        """
        CB might be able to provide additional information from the dataset about the state
        This property is used by the VerbalContextualBandit
        :return:
        """
        raise NotImplementedError


# the step method can be written abstractly (because it just calls core_bandit)
class VerbalContextualBandit(ContextualBandit):
    def __init__(self, core_bandit, *args, **kwargs):
        self.core_bandit = core_bandit

    @property
    def name(self) -> str:
        # cb_1m-ratings_arms10
        return self.core_bandit.name

