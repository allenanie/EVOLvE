from typing import Dict, Any, Tuple, Union, List, Optional
import numpy as np

Action = Union[int, str]
ExpectedReward = float


class Bandit:
    num_arms: int
    horizon: int
    seed: Optional[int]
    h: int

    def initialize_defaults(self) -> None:
        raise NotImplementedError

    def reset(self) -> Tuple[None, Dict[str, Any]]:
        raise NotImplementedError

    def set_seed(self, seed: Optional[int] = None) -> None:
        self.seed = seed
        self.np_random = np.random.default_rng(self.seed)

    @property
    def name(self) -> str:
        # b_vid_arms5_easy
        raise NotImplementedError
