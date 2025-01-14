from typing import List, Dict, Any, Union
import scipy
import numpy as np
from pydantic import BaseModel
from banditbench.agents.classics import MABAgent, CBAgent, UCBAgent, ThompsonSamplingAgent, GreedyAgent, LinUCBAgent
from banditbench.tasks.cb.env import State


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

    def to_str(self):
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

    def to_str(self):
        return str(self)

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


class VerbalGuide:
    # VerbalGuide can be retrieved in two ways:
    # it's a verbal analog of a Q-function
    # Q(s=None, a) # MAB
    # Q(s, a) # CB

    def __init__(self, agent: Union[MABAgent, CBAgent]):
        self.agent = agent

    def get_action_guide_info(self, arm: int) -> ActionInfo:
        raise NotImplementedError("Only for MAB agents")

    def get_actions_guide_info(self) -> List[ActionInfo]:
        raise NotImplementedError("Only for MAB agents")

    def get_state_action_guide_info(self, state: State, arm: int) -> ActionInfo:
        raise NotImplementedError("Only for RL and CB agents")

    def get_state_actions_guide_info(self, state: State) -> List[ActionInfo]:
        raise NotImplementedError("Only for RL and CB agents")


class UCBGuide(VerbalGuide):
    # takes in UCBAgent and then return info on each arm (a block of text)
    def __init__(self, agent: UCBAgent):
        super().__init__(agent)

    def get_actions_guide_info(self) -> List[ActionInfo]:
        actions_info = []
        for arm in range(self.agent.k_arms):
            arm_info = self.get_state_action_guide_info(arm)
            actions_info.append(arm_info)

        assert len(actions_info) == self.agent.k_arms
        return actions_info

    def get_action_guide_info(self, arm: int) -> ActionInfo:
        exploration_bonus = self.agent.calculate_exp_bonus(arm) if self.agent.arms[arm] > 0 else "inf"
        exp_bonus_guide = ActionInfoField(info_name='exploration bonus', value=exploration_bonus)

        exploitation_value = self.agent.calculate_exp_value(arm) if self.agent.arms[arm] > 0 else 0
        exp_value_guide = ActionInfoField(info_name='exploitation value', value=exploitation_value)
        return exp_bonus_guide + exp_value_guide


class GreedyGuide(VerbalGuide):
    def __init__(self, agent: GreedyAgent):
        super().__init__(agent)

    def get_actions_guide_info(self) -> List[ActionInfo]:
        actions_info = []
        for arm in range(self.agent.k_arms):
            arm_info = self.get_action_guide_info(arm)
            actions_info.append(arm_info)
        return actions_info

    def get_action_guide_info(self, arm: int) -> ActionInfo:
        exploitation_value = self.agent.calculate_exp_value(arm) if self.agent.arms[arm] > 0 else 0
        exp_value_guide = ActionInfoField(info_name='exploitation value', value=exploitation_value)
        arm_info = ActionInfo(action_info_fields=[exp_value_guide])
        return arm_info


class ThompsonSamplingGuide(VerbalGuide):
    def __init__(self, agent: ThompsonSamplingAgent):
        super().__init__(agent)

    def get_guide_info(self) -> List[ActionInfo]:
        actions_info = []
        for arm in range(self.agent.k_arms):
            arm_info = self.get_action_guide_info(arm)
            actions_info.append(arm_info)

        assert len(actions_info) == len(self.agent.actions)
        return actions_info

    def get_action_guide_info(self, arm: int) -> ActionInfo:
        alpha = self.agent.alpha[arm]
        beta = self.agent.beta[arm]
        p = scipy.stats.beta.rvs(self.agent.alpha[arm], self.agent.beta[arm])
        alpha_guide = ActionInfoField(info_name='alpha', value=alpha,
                                      info_template='prior beta distribution(alpha={:.2f}')
        beta_guide = ActionInfoField(info_name='beta', value=beta, info_template='beta={:.2f})')
        probability_guide = ActionInfoField(info_name='probability', value=p,
                                            info_template='posterior bernoulli p={:.2f}')
        return alpha_guide + beta_guide + probability_guide


class LinUCBGuide(VerbalGuide):
    def __init__(self, agent: LinUCBAgent):
        super().__init__(agent)

    def get_state_action_guide_info(self, state: State, arm: int) -> ActionInfo:
        a = arm
        context = state.feature

        A_inv = np.linalg.inv(self.agent.A[a])
        theta = A_inv.dot(self.agent.b[a])
        exploration_bonus = self.agent.alpha * np.sqrt(context.T.dot(A_inv).dot(context))
        exploitation_value = theta.T.dot(context)[0, 0]

        exp_bonus_guide = ActionInfoField(info_name='exploration bonus', value=exploration_bonus)
        exp_value_guide = ActionInfoField(info_name='exploitation value', value=exploitation_value)

        return exp_bonus_guide + exp_value_guide

    def get_state_actions_guide_info(self, state: State) -> List[ActionInfo]:
        actions_info = []
        for arm in range(self.agent.k_arms):
            arm_info = self.get_state_action_guide_info(state, arm)
            actions_info.append(arm_info)

        return actions_info
