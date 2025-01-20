import numpy as np
from typing import Union, List, Optional
from banditbench.tasks.types import State

class BanditScenario:
    name: str

    action_names: List[str]
    action_unit: str

    base_description: str
    detailed_description: str
    query_prompt: str = ("\n\nWhich {unit} will you choose next? PLEASE RESPOND ONLY WITH {choices} AND NO TEXT "
                         "EXPLANATION.")

    fewshot_prompt: str = (
                "Here are some examples of optimal actions under different scenarios."
                " Use them as hints to help you come up with better actions.\n"
            )

    def __init__(self, num_actions: int,
                 action_names: List[str], action_unit: str,
                 base_description: str, detailed_description: str,
                 query_prompt: str,
                 seed: Union[int, None] = None):

        self.action_names = action_names
        self.action_unit = action_unit
        self.base_description = base_description
        self.detailed_description = detailed_description
        self.query_prompt = query_prompt

        if seed is not None:
            self.set_seed(seed)
            self.np_random.shuffle(self.action_names)
            self.action_names = self.action_names[:num_actions]
        else:
            self.action_names = self.action_names[:num_actions]

    def set_seed(self, seed: Optional[int] = None) -> None:
        self.seed = seed
        self.np_random = np.random.default_rng(self.seed)

    def get_instruction(self, version="base") -> str:
        raise NotImplementedError


class MABScenario(BanditScenario):
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

        return prompt

    def get_query_prompt(self) -> str:
        prompt = self.query_prompt.format(unit=self.action_unit, choices="[" + ", ".join(self.action_names) + "]")
        return prompt


class CBScenario(BanditScenario):
    def get_query_prompt(self, state: State, side_info: Optional[str] = None) -> str:
        """For contextual bandit, the agent can pass in optional side_info to the decision query context"""
        raise NotImplementedError
