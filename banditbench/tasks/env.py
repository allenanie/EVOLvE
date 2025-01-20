from typing import Dict, Any, Tuple, Union, List, Optional
import json
import numpy as np
from banditbench.tasks.scenario import BanditScenario
from banditbench.tasks.types import Action


class Bandit:
    num_arms: int
    horizon: int
    seed: Optional[int]
    h: int  # number of interactions

    def initialize_defaults(self) -> None:
        raise NotImplementedError

    def reset(self, seed: Optional[int] = None) -> Tuple[None, Dict[str, Any]]:
        raise NotImplementedError

    def set_seed(self, seed: Optional[int] = None) -> None:
        self.seed = seed
        self.np_random = np.random.default_rng(self.seed)

    @property
    def name(self) -> str:
        # b_vid_arms5_easy
        raise NotImplementedError


class VerbalBandit(Bandit):
    bandit_scenario: BanditScenario

    def __init__(self, core_bandit, *args, **kwargs):
        self.core_bandit = core_bandit
        self.num_arms = self.core_bandit.num_arms
        self.horizon = self.core_bandit.horizon
        self.seed = self.core_bandit.seed
        self.h = self.core_bandit.h

    def verbalize_feedback(self, action_name: Action, reward: float) -> str:
        """This corresponds to raw / unprocessed feedback from the environment"""
        raise NotImplementedError

    def verbalize_state(self, observation: Any) -> str:
        raise NotImplementedError

    @property
    def action_names(self) -> List[str]:
        raise NotImplementedError

    def get_query_prompt(self, *args, **kwargs) -> str:
        raise NotImplementedError

    def get_task_instruction(self, *args, **kwargs) -> str:
        raise NotImplementedError


# this is for serialization and for string translation
