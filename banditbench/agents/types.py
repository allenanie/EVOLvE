from typing import Union, Dict, Any, List
from banditbench.tasks.env import Bandit, VerbalBandit
from banditbench.tasks.cb.env import State

from pydantic import BaseModel


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


class ActionInfoField(BaseModel):
    info_name: str  # such as "exploitation value"
    info_template: Union[str, None]  # Note, we only support non-key-value templates, like "value name = {:.2f}"
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

    def to_str(self, json_fmt=False):
        """
        :param json_fmt: if set True, will mimic JSON format in string like `{info_name: value}` explicitly in string
        :return:
        """
        if json_fmt and self.info_template is None:
            if isinstance(self.value, float):
                return "{" + f"\"{self.info_name}\": {self.value:.2f}" + "}"
            else:
                return "{" + f"\"{self.info_name}\": {self.value}" + "}"
        else:
            return str(self)

    def __add__(self, other: Union['ActionInfoField', 'ActionInfo']):
        if isinstance(other, ActionInfoField):
            return ActionInfo(action_info_fields=[self, other])
        elif isinstance(other, ActionInfo):
            return ActionInfo(action_info_fields=[self] + other.action_info_fields)
        else:
            raise ValueError(f"Unsupported type: {type(other)}")


class ActionInfo(BaseModel):
    # an action can have multiple fields (of information)
    action_info_fields: List[ActionInfoField]

    def __str__(self):
        return ", ".join([info.to_str() for info in self.action_info_fields])

    def to_str(self, json_fmt=False):
        if not json_fmt:
            return str(self)
        else:
            return ", ".join([info.to_str(json_fmt=True) for info in self.action_info_fields])

    def __len__(self):
        return len(self.action_infos)

    def __add__(self, other: Union['ActionInfo', 'ActionInfoField']):
        if isinstance(other, ActionInfoField):
            return ActionInfo(action_info_fields=self.action_info_fields + [other])
        elif isinstance(other, ActionInfo):
            return ActionInfo(action_info_fields=self.action_info_fields + other.action_info_fields)
        else:
            raise ValueError(f"Unsupported type: {type(other)}")

    def get_info_by_name(self, info_name: str) -> Union[ActionInfoField, None]:
        """Retrieve an ActionInfoField by its info_name."""
        for field in self.action_info_fields:
            if field.info_name == info_name:
                return field
        return None

    def get_value_by_name(self, info_name: str) -> Union[float, str, None]:
        """Retrieve just the value of an ActionInfoField by its info_name."""
        field = self.get_info_by_name(info_name)
        if field is not None:
            return field.value
        return None

    def __getitem__(self, key: Union[str, int]) -> Union[float, str, None, ActionInfoField]:
        """Allow both dictionary-style access by info_name and list-style access by index."""
        if isinstance(key, str):
            return self.get_value_by_name(key)
        elif isinstance(key, int):
            if 0 <= key < len(self.action_info_fields):
                return self.action_info_fields[key]
            raise IndexError("Index out of range")
        raise TypeError(f"Invalid key type: {type(key)}")
