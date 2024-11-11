from typing import List, Dict, Any, Union
import scipy
from pydantic import BaseModel
from banditbench.algs.classics import Agent, UCBAgent, ThompsonSamplingAgent, GreedyAgent


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
            return ActionInfo(action_info_fields=self.action_infos + other.action_infos)
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


class VerbalGuide:
    # VerbalGuide can be retrieved in two ways:
    # it's a verbal analog of a Q-function
    # Q(s=None, a) # MAB
    # Q(s, a) # CB

    def __init__(self, agent: Agent):
        self.agent = agent

    def get_action_guide_info(self, arm: int) -> ActionInfo:
        raise NotImplementedError("Only for MAB agents")

    def get_actions_guide_info(self) -> List[ActionInfo]:
        raise NotImplementedError("Only for MAB agents")

    def get_state_action_guide_info(self, arm: int) -> ActionInfo:
        raise NotImplementedError("Only for RL and CB agents")

    def get_state_actions_guide_info(self) -> List[ActionInfo]:
        raise NotImplementedError("Only for RL and CB agents")


class UCBGuide(VerbalGuide):
    # takes in UCBAgent and then return info on each arm (a block of text)
    def __init__(self, agent: UCBAgent):
        super().__init__(agent)

    def get_guide_info(self) -> List[ActionInfo]:
        actions_info = []
        for arm in range(self.agent.k_arms):
            exploration_bonus = self.agent.calculate_exp_bonus(arm) if self.agent.arms[arm] > 0 else "inf"
            exp_bonus_guide = ActionInfoField(info_name='exploration_bonus', value=exploration_bonus)

            exploitation_value = self.agent.calculate_exp_value(arm) if self.agent.arms[arm] > 0 else 0
            exp_value_guide = ActionInfoField(info_name='exploitation_value', value=exploitation_value)

            actions_info.append(exp_bonus_guide + exp_value_guide)

        assert len(actions_info) == len(self.agent.actions)
        return actions_info

    def get_action_guide_info(self, arm: int) -> ActionInfo:
        exploration_bonus = self.agent.calculate_exp_bonus(arm) if self.agent.arms[arm] > 0 else "inf"
        exp_bonus_guide = ActionInfoField(info_name='exploration_bonus', value=exploration_bonus)

        exploitation_value = self.agent.calculate_exp_value(arm) if self.agent.arms[arm] > 0 else 0
        exp_value_guide = ActionInfoField(info_name='exploitation_value', value=exploitation_value)
        return exp_bonus_guide + exp_value_guide

class GreedyGuide(VerbalGuide):
    def __init__(self, agent: GreedyAgent):
        super().__init__(agent)

    def get_guide_info(self) -> List[ActionInfo]:
        actions_info = []
        for arm in range(self.agent.k_arms):
            exploitation_value = self.agent.calculate_exp_value(arm) if self.agent.arms[arm] > 0 else 0
            exp_value_guide = ActionInfoField(info_name='exploitation_value', value=exploitation_value)
            actions_info.append(exp_value_guide)
        return actions_info
    
    def get_action_guide_info(self, arm: int) -> ActionInfo:
        exploitation_value = self.agent.calculate_exp_value(arm) if self.agent.arms[arm] > 0 else 0
        exp_value_guide = ActionInfoField(info_name='exploitation_value', value=exploitation_value)
        return exp_value_guide


class ThompsonSamplingGuide(VerbalGuide):
    def __init__(self, agent: ThompsonSamplingAgent):
        super().__init__(agent)

    def get_guide_info(self) -> List[ActionInfo]:
        actions_info = []
        for arm in range(self.agent.k_arms):
            alpha = self.agent.alpha[arm]
            beta = self.agent.beta[arm]
            p = scipy.stats.beta.rvs(self.agent.alpha[arm], self.agent.beta[arm])
            alpha_guide = ActionInfoField(info_name='alpha', value=alpha,
                                          info_template='Prior Beta Distribution(alpha={:.2f}')
            beta_guide = ActionInfoField(info_name='beta', value=beta, info_template='beta={:.2f})')
            probability_guide = ActionInfoField(info_name='probability', value=p,
                                                info_template='Posterior Bernoulli p={:.2f}')

            actions_info.append(alpha_guide + beta_guide + probability_guide)

        assert len(actions_info) == len(self.agent.actions)
        return actions_info

    def get_action_guide_info(self, arm: int) -> ActionInfo:
        alpha = self.agent.alpha[arm]
        beta = self.agent.beta[arm]
        p = scipy.stats.beta.rvs(self.agent.alpha[arm], self.agent.beta[arm])
        alpha_guide = ActionInfoField(info_name='alpha', value=alpha,
                                      info_template='Prior Beta Distribution(alpha={:.2f}')
        beta_guide = ActionInfoField(info_name='beta', value=beta, info_template='beta={:.2f})')
        probability_guide = ActionInfoField(info_name='probability', value=p,
                                            info_template='Posterior Bernoulli p={:.2f}')
        return alpha_guide + beta_guide + probability_guide
