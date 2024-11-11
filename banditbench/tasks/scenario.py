import os
import numpy as np
from pydantic import BaseModel
from typing import Dict, Any, Tuple, Union, List, Callable, Set, Optional
from banditbench.tasks.utils import dedent


class BanditConfig(BaseModel):
    bandit_type: str
    domain: str
    difficulty: str
    num_arms: int

    def get_file_name(self):
        return f"bandit_{self.bandit_type}_{self.domain}_arms{self.num_arms}_{self.difficulty}_trial0_fewshot_equal_space.json"

    def get_file_path(self, base_dir: str):
        return os.path.join(base_dir, self.get_file_name())


class BanditScenario:
    action_names: List[str]
    action_unit: str

    base_description: str
    detailed_description: str

    num_fewshot: int
    fewshot_examples: str
    few_shot_config: Optional[BanditConfig]

    def __init__(self, num_actions: int,
                 action_names: List[str], action_unit: str,
                 base_description: str, detailed_description: str,
                 seed: Union[int, None] = None,
                 num_fewshot: int = 0, few_shot_config: Optional[BanditConfig] = None):

        self.action_names = action_names
        self.action_unit = action_unit
        self.base_description = base_description
        self.detailed_description = detailed_description
        self.num_fewshot = num_fewshot
        self.few_shot_config = few_shot_config
        self.initialize_defaults()

        if seed is not None:
            self.set_seed(seed)
            self.np_random.shuffle(self.action_names)
            self.action_names = self.action_names[:num_actions]
        else:
            self.action_names = self.action_names[:num_actions]

        if num_fewshot > 0:
            assert few_shot_config is not None, "Few-shot config must be provided"
            self.fewshot_examples = self.load_fewshot_examples()

    def initialize_defaults(self) -> None:
        self.fewshot_examples = ""

    def set_seed(self, seed: Optional[int] = None) -> None:
        self.seed = seed
        self.np_random = np.random.default_rng(self.seed)

    def get_instruction(self, version="base"):
        """We add the few-shot examples in here"""
        if version == "base":
            prompt = self.base_description.format(
                len(self.action_names), "[" + ", ".join(self.action_names) + "]"
            )
        elif version == "detailed":
            prompt = self.detailed_description.format(
                len(self.action_names), "[" + ", ".join(self.action_names) + "]"
            )
        else:
            raise ValueError(f"Unknown description version {version}")

        if self.fewshot_examples != "":
            prompt += "\n" + self.fewshot_examples

        return prompt

    def load_fewshot_examples(self):
        """Few-shot examples have their own configuration such as the number of arms, difficulty, scenario, etc."""
        # reminder: you need to implement this later
        raise NotImplementedError
        return ""