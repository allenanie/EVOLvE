from typing import Optional, List, Any, Dict, Union

import litellm
from banditbench.agents.classics import MABAgent, CBAgent
from banditbench.agents.guides import VerbalGuide
from banditbench.tasks.typing import State
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
    history: List[Union[mab.VerbalInteraction, cb.VerbalInteraction]]
    history_context_len: int

    def __init__(self, history_context_len: int):
        self.history = []
        self.history_context_len = history_context_len

    def _represent_interaction_history(self, action_names: List[str], action_unit: str,
                                       history_len: int) -> str:
        """Get formatted history for LLM prompt."""
        # Implement history formatting
        raise NotImplementedError


class RawHistoryFunc(HistoryFunc):
    """Formats raw interaction history for LLM prompt."""

    def _represent_interaction_history(self, action_names: List[str], action_unit: str,
                                       history_len: int) -> str:
        if len(self.history) == 0:
            return ""

        # remember to handle state
        history_len = min(history_len, len(self.history))
        snippet = ""
        for exp in self.history[-history_len:]:
            snippet += f"\n{exp.feedback}"

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
        if len(self.history) == 0:
            return ""

        # compute basic statistics, for each action name
        # frequency, mean reward

        n_actions = [0] * len(action_names)
        action_rewards = [0] * len(action_names)

        for exp in self.history:
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


class AlgorithmGuideFunc(HistoryFunc):
    """Provides algorithm guidance text for LLM prompt."""
    pass


class LLMMABAgent(MABAgent, LLM, HistoryFunc):
    """LLM-based multi-armed bandit agent."""

    history: List[mab.VerbalInteraction]

    decision_context_start: str = "So far you have interacted {} times with the following choices and rewards:\n"

    def __init__(self, env: VerbalBandit,
                 ag: VerbalGuide = None,
                 model: str = "gpt-3.5-turbo",
                 history_context_len=1000,
                 verbose=False):
        MABAgent.__init__(self, env)
        LLM.__init__(self, model)
        HistoryFunc.__init__(self, history_context_len)
        self.ag = ag
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
        self.history.append(info['interaction'])

    def represent_history(self):
        return self._represent_interaction_history(self.env.action_names, self.env.bandit_scenario.action_unit,
                                                   self.history_context_len)


class LLMCBAgent(CBAgent, LLM, HistoryFunc):
    """LLM-based contextual bandit agent."""

    history: List[cb.VerbalInteraction]

    decision_context_start: str = ("So far you have interacted {} times with the most recent following choices and "
                                   "rewards:\n")

    def __init__(self, env: VerbalBandit,
                 ag: VerbalGuide = None,
                 model: str = "gpt-3.5-turbo",
                 history_context_len=1000,
                 verbose=False):
        CBAgent.__init__(self, env)
        LLM.__init__(self, model)
        HistoryFunc.__init__(self, history_context_len)
        self.ag = ag
        self.verbose = verbose

    def act(self, state: State) -> str:
        """Generate next action using LLM and context."""
        # Implement LLM-based contextual action selection
        task_instruction = self.env.get_task_instruction()
        history_context = self.represent_history()
        # side_info = self.get_guidance(state) if self.ag is not None else None
        query = self.env.get_query_prompt(state, None)

        response = self.generate(task_instruction + history_context + query)
        return response

    def update(self, state: State, action: int, reward: float, info: Dict[str, Any]) -> None:
        assert 'interaction' in info
        assert type(info['interaction']) is cb.VerbalInteraction
        self.history.append(info['interaction'])

    def represent_history(self):
        return self._represent_interaction_history(self.env.action_names, self.env.bandit_scenario.action_unit,
                                                   self.history_context_len)
