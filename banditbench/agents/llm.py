from typing import Optional, List, Any, Dict, Union

import litellm
from banditbench.agents.classics import MABAgent, CBAgent
from banditbench.agents.guides import VerbalGuide, UCBGuide, LinUCBGuide, ActionInfo
from banditbench.tasks.typing import State, Info
from banditbench.tasks.env import VerbalBandit

import banditbench.tasks.cb as cb
import banditbench.tasks.mab as mab


class LLM:
    """Base class for LLM functionality shared across agent types."""

    def __init__(self, model: str = "gpt-3.5-turbo",
                 api_base: Optional[str] = None):
        """Initialize LLM agent with specified model.
        
        Args:
            model: Name of LLM model to use (default: gpt-3.5-turbo)
        """
        self.model = model

    def generate(self, message: str) -> str:
        """Generate LLM response for given messages.

        Returns:
            Generated response text
        """
        print("Message to LLM")
        print(message)
        return "Mock LLM Response"
        # response = litellm.completion(
        #     model=self.model,
        #     messages=[{"content": message, "role": "user"}]
        # )
        # return response.choices[0].message.content


class HistoryFunc:
    interaction_history: List[Union[mab.VerbalInteraction, cb.VerbalInteraction]]
    history_context_len: int

    def __init__(self, history_context_len: int):
        self.interaction_history = []
        self.history_context_len = history_context_len

    def _represent_interaction_history(self, action_names: List[str], action_unit: str,
                                       history_len: int) -> str:
        """Get formatted history for LLM prompt."""
        # Implement history formatting
        raise NotImplementedError

    def reset(self):
        self.interaction_history = []


class MABRawHistoryFunc(HistoryFunc):
    """Formats raw interaction history for LLM prompt."""

    def _represent_interaction_history(self, action_names: List[str], action_unit: str,
                                       history_len: int) -> str:
        if len(self.interaction_history) == 0:
            return ""

        # remember to handle state
        history_len = min(history_len, len(self.interaction_history))
        snippet = ""
        for exp in self.interaction_history[-history_len:]:
            snippet += f"\n{exp.feedback}"  # MAB feedback contains {action_name} {reward} already

        return snippet

class CBRawHistoryFunc(HistoryFunc):
    def _represent_interaction_history(self, action_names: List[str], action_unit: str,
                                       history_len: int) -> str:
        if len(self.interaction_history) == 0:
            return ""

        # remember to handle state
        history_len = min(history_len, len(self.interaction_history))
        snippet = ""
        for exp in self.interaction_history[-history_len:]:
            snippet += f"\nContext: {exp.feedback}"
            snippet += f"\nAction: {exp.mapped_action_name.split(') (')[0] + ')'}"  # this is to replicate the same style as the paper
            snippet += f"\nReward: {exp.reward}\n"

        return snippet

class SummaryHistoryFunc(HistoryFunc):
    """Summarizes interaction history for LLM prompt."""

    def _represent_interaction_history(self, action_names: List[str], action_unit: str,
                                       history_len: int) -> str:
        """
        Note that this function can work with either MAB or CB
        But for CB, it is not summarizing on the state level
        """
        # we traverse through the whole history to summarize
        if len(self.interaction_history) == 0:
            return ""

        # compute basic statistics, for each action name
        # frequency, mean reward

        n_actions = [0] * len(action_names)
        action_rewards = [0] * len(action_names)

        for exp in self.interaction_history:
            idx = action_names.index(exp.mapped_action_name)
            n_actions[idx] += 1
            action_rewards[idx] += exp.reward

        snippet = ""
        for action_name, n, total_r in zip(action_names, n_actions, action_rewards):
            reward = total_r / (n + 1e-6)
            snippet += (
                f"\n{action_name} {action_unit}, {n} times, average"
                f" reward {reward:.2f}"
            )

        return snippet


class LLMMABAgent(MABAgent, LLM, HistoryFunc):
    """LLM-based multi-armed bandit agent."""

    interaction_history: List[mab.VerbalInteraction]

    decision_context_start: str = "So far you have interacted {} times with the following choices and rewards:\n"

    def __init__(self, env: VerbalBandit,
                 model: str = "gpt-3.5-turbo",
                 history_context_len=1000,
                 verbose=False):
        MABAgent.__init__(self, env)
        LLM.__init__(self, model)
        HistoryFunc.__init__(self, history_context_len)
        self.verbose = verbose

    def act(self) -> str:
        """Generate next action using LLM."""
        # Implement LLM-based action selection
        task_instruction = self.env.get_task_instruction()
        history_context = self.represent_history()
        query = self.env.get_query_prompt()

        response = self.generate(task_instruction + history_context + query)
        return response

    def update(self, action: int, reward: float, info: Dict[str, Any]) -> None:
        # note that we don't have access to the expected reward on the agent side
        assert 'interaction' in info
        assert type(info['interaction']) is mab.VerbalInteraction
        self.interaction_history.append(info['interaction'])

    def represent_history(self):
        return self._represent_interaction_history(self.env.action_names, self.env.bandit_scenario.action_unit,
                                                   self.history_context_len)

    def reset(self):
        super().reset()


class LLMCBAgent(CBAgent, LLM, HistoryFunc):
    """LLM-based contextual bandit agent."""

    interaction_history: List[cb.VerbalInteraction]

    decision_context_start: str = ("So far you have interacted {} times with the most recent following choices and "
                                   "rewards:\n")

    def __init__(self, env: VerbalBandit,
                 model: str = "gpt-3.5-turbo",
                 history_context_len=1000,
                 verbose=False):
        CBAgent.__init__(self, env)
        LLM.__init__(self, model)
        HistoryFunc.__init__(self, history_context_len)
        self.verbose = verbose

    def act(self, state: State) -> str:
        """Generate next action using LLM and context."""
        # Implement LLM-based contextual action selection
        task_instruction = self.env.get_task_instruction()
        history_context = self.represent_history()
        query = self.env.get_query_prompt(state, None)

        response = self.generate(task_instruction + history_context + query)
        return response

    def update(self, state: State, action: int, reward: float, info: Dict[str, Any]) -> None:
        assert 'interaction' in info
        assert type(info['interaction']) is cb.VerbalInteraction
        self.interaction_history.append(info['interaction'])

    def represent_history(self):
        return self._represent_interaction_history(self.env.action_names, self.env.bandit_scenario.action_unit,
                                                   self.history_context_len)

    def reset(self):
        super().reset()


class LLMMABAgentSH(LLMMABAgent, SummaryHistoryFunc):
    # MAB SH Agent
    ...


class LLMMABAgentRH(LLMMABAgent, MABRawHistoryFunc):
    # MAB RH Agent
    ...


class LLMCBAgentRH(LLMCBAgent, CBRawHistoryFunc):
    # CB RH Agent
    ...


class MABAlgorithmGuideWithSummaryHistory(HistoryFunc):
    """Provides algorithm guidance text for LLM prompt."""
    ag: UCBGuide

    def __init__(self, ag: UCBGuide, history_context_len: int):
        super().__init__(history_context_len)
        self.ag = ag
        assert type(ag) is UCBGuide, "Only UCBGuide works with SummaryHistory -- since the summary is per action level."

    def update_algorithm_guide(self, action: int, reward: float, info: Dict[str, Any]) -> None:
        """Enhance update to include algorithm guide updates."""
        # First call the parent class's update
        self.ag.agent.update(action, reward, info)

    def reset(self):
        self.ag.agent.reset()

    def _represent_interaction_history(self, action_names: List[str], action_unit: str,
                                       history_len: int) -> str:
        """
        Note that this function can work with either MAB or CB
        But for CB, it is not summarizing on the state level
        """
        # we traverse through the whole history to summarize
        if len(self.interaction_history) == 0:
            return ""

        n_actions = [0] * len(action_names)
        action_rewards = [0] * len(action_names)

        for exp in self.interaction_history:
            idx = action_names.index(exp.mapped_action_name)
            n_actions[idx] += 1
            action_rewards[idx] += exp.reward

        snippet, action_idx = "", 0
        for action_name, n, total_r in zip(action_names, n_actions, action_rewards):
            reward = total_r / (n + 1e-6)
            snippet += (
                    f"\n{action_name} {action_unit}, {n} times, average"
                    f" reward {reward:.2f}" + " " + self.ag.get_action_guide_info(action_idx).to_str()
            )
            action_idx += 1

        return snippet


class CBAlgorithmGuideWithRawHistory(HistoryFunc):
    """Provides algorithm guidance text for LLM prompt."""
    ag: LinUCBGuide
    ag_info_history: List[List[ActionInfo]]  # storing side information

    def __init__(self, ag: LinUCBGuide, history_context_len: int):
        super().__init__(history_context_len)
        self.ag_info_history = []
        self.ag = ag
        assert type(ag) is LinUCBGuide, "The information is provided per context, per action"

    def reset(self):
        self.ag.agent.reset()

    def update_algorithm_guide(self, state: State, action: int, reward: float, info: Dict[str, Any]) -> None:
        """Enhance update to include algorithm guide updates."""
        # First call the parent class's update
        self.ag.agent.update(state, action, reward, info)

    def update_info_history(self, action_info: List[ActionInfo]) -> None:
        self.ag_info_history.append(action_info)

    def _represent_interaction_history(self, action_names: List[str], action_unit: str,
                                       history_len: int) -> str:
        if len(self.interaction_history) == 0:
            return ""

        # remember to handle state
        history_len = min(history_len, len(self.interaction_history))
        snippet = ""
        for exp, ag_info in zip(self.interaction_history[-history_len:], self.ag_info_history[-history_len:]):
            snippet += f"\nContext: {exp.feedback}"
            snippet += f"\nSide Information for decision making:"
            for action_info in ag_info:
                snippet += '\n' + action_info.to_str()
            snippet += f"\nAction: {exp.mapped_action_name.split(') (')[0] + ')'}"  # this is to replicate the same style as the paper
            snippet += f"\nReward: {exp.reward}\n"

        return snippet


class LLMMABAgentSHWithAG(LLMMABAgent, LLM, MABAlgorithmGuideWithSummaryHistory):
    def __init__(self, env: VerbalBandit,
                 ag: UCBGuide,
                 model: str = "gpt-3.5-turbo",
                 history_context_len=1000,
                 verbose=False):
        MABAgent.__init__(self, env)
        LLM.__init__(self, model)
        MABAlgorithmGuideWithSummaryHistory.__init__(self, ag,
                                                     history_context_len)
        self.verbose = verbose

    def update(self, action: int, reward: float, info: Dict[str, Any]) -> None:
        # note that we don't have access to the expected reward on the agent side
        super().update(action, reward, info)
        self.update_algorithm_guide(action, reward, info)

    def reset(self):
        super().reset()


class LLMCBAgentRHWithAG(LLMCBAgent, LLM, CBAlgorithmGuideWithRawHistory):
    def __init__(self, env: VerbalBandit,
                 ag: LinUCBGuide,
                 model: str = "gpt-3.5-turbo",
                 history_context_len=1000,
                 verbose=False):
        MABAgent.__init__(self, env)
        LLM.__init__(self, model)
        CBAlgorithmGuideWithRawHistory.__init__(self, ag,
                                                history_context_len)
        self.verbose = verbose

    def update(self, state: State, action: int, reward: float, info: Dict[str, Any]) -> None:
        # note that we don't have access to the expected reward on the agent side
        super().update(state, action, reward, info)
        # store side information; this is the information we used to make the decision (because algorithm guide has
        # not been updated yet)
        self.update_info_history(self.ag.get_state_actions_guide_info(state))
        self.update_algorithm_guide(state, action, reward, info)

    def reset(self):
        super().reset()
