from typing import Optional, List, Any

import litellm
from banditbench.agents.classics import MABAgent, CBAgent
from banditbench.agents.guides import VerbalGuide
from banditbench.tasks.cb.env import State
from banditbench.tasks.cb.env import Interaction as CBInteraction
from banditbench.tasks.mab.env import Interaction as MABInteraction
from banditbench.tasks.env import VerbalBandit


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
        response = litellm.completion(
            model=self.model,
            messages=[{"content": message, "role": "user"}]
        )
        return response.choices[0].message.content


class HistoryFunc:
    history: List[Any]

    def get_interaction_history(self) -> str:
        """Get formatted history for LLM prompt."""
        # Implement history formatting
        raise NotImplementedError


class RawHistoryFunc(HistoryFunc):
    """Formats raw interaction history for LLM prompt."""
    pass


class SummaryHistoryFunc(HistoryFunc):
    """Summarizes interaction history for LLM prompt."""
    pass


class AlgorithmGuideFunc(HistoryFunc):
    """Provides algorithm guidance text for LLM prompt."""
    pass


class LLMMABAgent(MABAgent, LLM, HistoryFunc):
    """LLM-based multi-armed bandit agent."""

    history: List[MABInteraction]

    decision_context_start: str = "So far you have interacted {} times with the following choices and rewards:\n"

    def __init__(self, env: VerbalBandit,
                 ag: VerbalGuide = None,
                 model: str = "gpt-3.5-turbo"):
        MABAgent.__init__(self, env)
        LLM.__init__(self, model)
        self.ag = ag

    def act(self) -> str:
        """Generate next action using LLM."""
        # Implement LLM-based action selection
        task_instruction = self.env.get_task_instruction()
        history_context = self.get_interaction_history()
        query = self.env.get_query_prompt()

        response = self.generate(task_instruction + history_context + query)
        return response

    def update(self, action: int, reward: float) -> None:
        pass


class LLMCBAgent(CBAgent, LLM, HistoryFunc):
    """LLM-based contextual bandit agent."""

    history: List[CBInteraction]

    decision_context_start: str = ("So far you have interacted {} times with the most recent following choices and "
                                   "rewards:\n")

    def __init__(self, env: VerbalBandit,
                 ag: VerbalGuide = None,
                 model: str = "gpt-3.5-turbo"):
        CBAgent.__init__(self, env)
        LLM.__init__(self, model)
        self.ag = ag

    def act(self, state: State) -> str:
        """Generate next action using LLM and context."""
        # Implement LLM-based contextual action selection
        task_instruction = self.env.get_task_instruction()
        history_context = self.get_interaction_history()
        side_info = self.get_guidance(state) if self.ag is not None else None
        query = self.env.get_query_prompt(state, side_info)

        response = self.generate(task_instruction + history_context + query)
        return response

    def update(self, state: State, action: int, reward: float) -> None:
        pass
