import json
import numpy as np
from pydantic import BaseModel, field_serializer
from typing import Dict, Any, Tuple, Union, List, Optional, Annotated

from banditbench.tasks.types import Action, InteractionBase, State, Info
from banditbench.tasks.env import Bandit



class Interaction(BaseModel, InteractionBase):
    state: State
    action: Action
    reward: float
    is_random: Union[bool, None] = None

    def __init__(self, state: State, action: Action, reward: float,
                 is_random: Union[bool, None] = None) -> None:
        super().__init__(state=state, action=action, reward=reward, is_random=is_random)


class VerbalInteraction(BaseModel, InteractionBase):
    state: State
    action: Action
    mapped_action: int  # index
    mapped_action_name: str
    reward: float
    feedback: str
    is_random: Union[bool, None] = None

    def __init__(self, state: State, action: Action, mapped_action: Action, mapped_action_name: Action,
                 reward: float, feedback: str, is_random: Union[bool, None] = None) -> None:
        super().__init__(state=state, action=action, mapped_action=mapped_action,
                         mapped_action_name=mapped_action_name,
                         reward=reward, feedback=feedback, is_random=is_random)


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

    def reset(self, seed: Optional[int] = None) -> Tuple[State, Info]:
        raise NotImplementedError

    def step(self, state: State, action: Action) -> Tuple[State, float, bool, Dict[str, Any]]:
        raise NotImplementedError

    @property
    def verbal_info(self) -> Dict[str, Any]:
        """
        CB might be able to provide additional information from the dataset about the state
        This property is used by the VerbalBandit
        :return:
        """
        raise NotImplementedError

    @property
    def feature_dim(self) -> int:
        """
        :return: dimension of the contextual feature space0.

        """
        raise NotImplementedError
