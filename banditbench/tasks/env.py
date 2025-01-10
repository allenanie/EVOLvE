from typing import Dict, Any, Tuple, Union, List, Optional
import json
import numpy as np
from pydantic import BaseModel

Action = Union[int, str]
ExpectedReward = float


class Bandit:
    num_arms: int
    horizon: int
    seed: Optional[int]
    h: int

    def initialize_defaults(self) -> None:
        raise NotImplementedError

    def reset(self, seed: Optional[int]=None) -> Tuple[None, Dict[str, Any]]:
        raise NotImplementedError

    def set_seed(self, seed: Optional[int] = None) -> None:
        self.seed = seed
        self.np_random = np.random.default_rng(self.seed)

    @property
    def name(self) -> str:
        # b_vid_arms5_easy
        raise NotImplementedError

class VerbalBandit(Bandit):

    def __init__(self, core_bandit, *args, **kwargs):
        self.core_bandit = core_bandit

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
class Trajectory(list):
    def __init__(self, interactions: Union[List['InteractionBase'], None] = None) -> None:
        super().__init__(interactions or [])

    def __add__(self, other: Union['InteractionBase', 'Trajectory']):
        if isinstance(other, InteractionBase):
            return Trajectory(list(self) + [other])
        elif isinstance(other, Trajectory):
            return Trajectory(list(self) + list(other))
        else:
            raise ValueError(f"Unsupported type: {type(other)}")

    def __getstate__(self):
        return list(self)

    def __setstate__(self, state):
        super().__init__(state)

    def __repr__(self) -> str:
        return f"Trajectory({super().__repr__()})"

    def model_dump(self, **kwargs) -> List[Dict[str, Any]]:
        return [
            item.model_dump(**kwargs) if hasattr(item, 'model_dump')
            else item.__dict__
            for item in self
        ]

    def model_dump_json(self, **kwargs) -> str:
        return json.dumps(self.model_dump(**kwargs))

class InteractionBase:
    def __add__(self, other: Union['InteractionBase', 'Trajectory']) -> Trajectory:
        if isinstance(other, InteractionBase):
            return Trajectory(interactions=[self, other])
        elif isinstance(other, Trajectory):
            return Trajectory(interactions=[self] + other.interactions)
        else:
            raise ValueError(f"Unsupported type: {type(other)}")
