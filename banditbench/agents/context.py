from typing import Optional, List, Any, Dict, Union

from banditbench.agents.guides import VerbalGuideBase, UCBGuide, LinUCBGuide, ActionInfo
from banditbench.tasks.types import State, Info

import banditbench.tasks.cb as cb
import banditbench.tasks.mab as mab

from banditbench.agents.types import MABAgent, CBAgent, ActionInfo


class ModelContextLayer:
    interaction_history: List[Union[mab.VerbalInteraction, cb.VerbalInteraction]]
    history_context_len: int

    def __init__(self, history_context_len: int):
        self.interaction_history = []
        self.history_context_len = history_context_len

    def represent_interaction_context(self, action_names: List[str], action_unit: str,
                                      history_len: int) -> str:
        """Get formatted history for LLM prompt."""
        # Implement history formatting
        raise NotImplementedError

    def reset(self):
        self.interaction_history = []


class RawContextLayerMAB(ModelContextLayer):
    """Formats raw interaction history for LLM prompt."""

    def represent_interaction_context(self, action_names: List[str], action_unit: str,
                                      history_len: int) -> str:
        if len(self.interaction_history) == 0:
            return ""

        # remember to handle state
        history_len = min(history_len, len(self.interaction_history))
        snippet = ""
        for exp in self.interaction_history[-history_len:]:
            snippet += f"\n{exp.feedback}"  # MAB feedback contains {action_name} {reward} already

        return snippet


class RawContextLayerCB(ModelContextLayer):
    def represent_interaction_context(self, action_names: List[str], action_unit: str,
                                      history_len: int) -> str:
        if len(self.interaction_history) == 0:
            return ""

        # remember to handle state
        history_len = min(history_len, len(self.interaction_history))
        snippet = ""
        for exp in self.interaction_history[-history_len:]:
            snippet += f"\nContext: {exp.state.feature_text}"
            snippet += f"\nAction: {exp.mapped_action_name}"  # this is to replicate the same style as the paper
            snippet += f"\nReward: {exp.reward}\n"

        return snippet


class SummaryContextLayerMAB(ModelContextLayer):
    """Summarizes interaction history for LLM prompt."""

    def represent_interaction_context(self, action_names: List[str], action_unit: str,
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


class SummaryAlgoGuideLayerMAB(SummaryContextLayerMAB):
    """Provides algorithm guidance text for LLM prompt."""
    ag: UCBGuide
    ag_info_history: List[List[ActionInfo]]  # storing side information

    def __init__(self, ag: UCBGuide, history_context_len: int):
        super().__init__(history_context_len)
        self.ag = ag
        self.ag_info_history = []
        assert type(ag) is UCBGuide, "Only UCBGuide works with SummaryHistory -- since the summary is per action level."

    def update_algorithm_guide(self, action: int, reward: float, info: Dict[str, Any]) -> None:
        """Enhance update to include algorithm guide updates."""
        # First call the parent class's update
        self.ag.agent.update(action, reward, info)

    def update_info_history(self, action_info: List[ActionInfo]) -> None:
        self.ag_info_history.append(action_info)

    def reset(self):
        super().reset()  # HistoryFunc.reset()
        self.ag.agent.reset()

    def represent_interaction_context(self, action_names: List[str], action_unit: str,
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


class RawContextAlgoGuideLayerCB(RawContextLayerCB):
    """Provides algorithm guidance text for LLM prompt."""
    ag: LinUCBGuide
    ag_info_history: List[List[ActionInfo]]  # storing side information

    def __init__(self, ag: LinUCBGuide, history_context_len: int):
        super().__init__(history_context_len)
        self.ag_info_history = []
        self.ag = ag
        assert type(ag) is LinUCBGuide, "The information is provided per context, per action"

    def reset(self):
        super().reset()  # HistoryFunc.reset()
        self.ag.agent.reset()
        self.ag_info_history = []

    def update_algorithm_guide(self, state: State, action: int, reward: float, info: Dict[str, Any]) -> None:
        """Enhance update to include algorithm guide updates."""
        # First call the parent class's update
        self.ag.agent.update(state, action, reward, info)

    def update_info_history(self, action_info: List[ActionInfo]) -> None:
        self.ag_info_history.append(action_info)

    def represent_interaction_context(self, action_names: List[str], action_unit: str,
                                      history_len: int) -> str:
        if len(self.interaction_history) == 0:
            return ""

        # remember to handle state
        history_len = min(history_len, len(self.interaction_history))
        snippet = ""
        for exp, ag_info in zip(self.interaction_history[-history_len:], self.ag_info_history[-history_len:]):
            snippet += f"\nContext: {exp.state.feature_text}"
            snippet += f"\nSide Information for decision making:"
            for i, action_info in enumerate(ag_info):
                # normal format
                # snippet += '\n' + action_names[i].split(") (")[0] + ")" + ": " + action_info.to_str()

                # JSON-like format used in the paper
                snippet += '\n{\"' + action_names[i] + ": " + action_info.to_str(
                    json_fmt=True) + "}"
            snippet += f"\nAction: {exp.mapped_action_name}"
            snippet += f"\nReward: {exp.reward}\n"

        return snippet
