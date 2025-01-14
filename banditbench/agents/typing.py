from typing import Union, Dict, Any
from banditbench.tasks.env import Bandit, VerbalBandit
from banditbench.tasks.cb.env import State

class Agent:
    name: str

    def __init__(self, env: Union[Bandit, VerbalBandit]) -> None:
        self.env = env
        self.k_arms = env.num_arms

    def reset(self):
        # no action
        pass


class MABAgent(Agent):

    def act(self) -> int:
        """Same as performing a sampling step."""
        raise NotImplementedError

    def update(self, action: int, reward: float, info: Dict[str, Any]) -> None:
        """The action performs an update step based on the action it chose, and the reward it received."""
        raise NotImplementedError


class CBAgent(Agent):
    def act(self, state: State) -> int:
        """Same as performing a sampling step."""
        raise NotImplementedError

    def update(self, state: State, action: int, reward: float, info: Dict[str, Any]) -> None:
        """The action performs an update step based on the action it chose, and the reward it received."""
        raise NotImplementedError
