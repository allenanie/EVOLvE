from typing import Optional, List, Any, Dict, Union

import litellm
from banditbench.agents.guides import VerbalGuide, UCBGuide, LinUCBGuide, ActionInfo
from banditbench.tasks.typing import State, Info
from banditbench.tasks.env import VerbalBandit
from banditbench.sampling.sampler import SampleWithLLMAgent

import banditbench.tasks.cb as cb
import banditbench.tasks.mab as mab

from banditbench.agents.typing import MABAgent, CBAgent


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
            snippet += f"\nContext: {exp.state.feature_text}"
            snippet += f"\nAction: {exp.mapped_action_name}"  # this is to replicate the same style as the paper
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
    demos: Optional[str]

    decision_context_start: str = "So far you have interacted {} times with the following choices and rewards:\n"

    def __init__(self, env: VerbalBandit,
                 model: str = "gpt-3.5-turbo",
                 history_context_len=1000,
                 verbose=False):
        MABAgent.__init__(self, env)
        LLM.__init__(self, model)
        HistoryFunc.__init__(self, history_context_len)
        self.demos = None  # few-shot demos, not reset, and only specified by FewShot class
        self.verbose = verbose

    def act(self) -> str:
        """Generate next action using LLM."""
        # Implement LLM-based action selection
        n_interactions = len(self.interaction_history)

        task_instruction = self.env.get_task_instruction()
        if self.demos is not None:
            task_instruction += self.demos + '\n'
        history_context = self.decision_context_start.format(n_interactions) + self.represent_history()
        query = self.env.get_query_prompt()

        response = self.generate(task_instruction + history_context + query)
        return response

    def update(self, action: int, reward: float, info: Dict[str, Any]) -> None:
        # note that we don't have access to the expected reward on the agent side
        assert 'interaction' in info
        assert type(info['interaction']) is mab.VerbalInteraction
        self.interaction_history.append(info['interaction'])

    def represent_history(self) -> str:
        return self._represent_interaction_history(self.env.action_names, self.env.bandit_scenario.action_unit,
                                                   self.history_context_len)

    def reset(self):
        super().reset()  # MABAgent.reset()
        self.interaction_history = []

    def get_task_instruction(self) -> str:
        task_instruction = self.env.get_task_instruction()
        return task_instruction

    def get_action_history(self) -> str:
        history_context = self.represent_history()
        return history_context

    def get_decision_query(self) -> str:
        query = self.env.get_query_prompt()
        return query


class LLMCBAgent(CBAgent, LLM, HistoryFunc):
    """LLM-based contextual bandit agent."""

    interaction_history: List[cb.VerbalInteraction]
    demos: Optional[str]  # few-shot demos, not reset, and only specified by FewShot class

    decision_context_start: str = ("So far you have interacted {} times with the most recent following choices and "
                                   "rewards:\n")

    def __init__(self, env: VerbalBandit,
                 model: str = "gpt-3.5-turbo",
                 history_context_len=1000,
                 verbose=False):
        CBAgent.__init__(self, env)
        LLM.__init__(self, model)
        HistoryFunc.__init__(self, history_context_len)
        self.demos = None
        self.verbose = verbose

    def act(self, state: State) -> str:
        """Generate next action using LLM and context."""
        # Implement LLM-based contextual action selection
        n_interactions = len(self.interaction_history)

        task_instruction = self.env.get_task_instruction()
        if self.demos is not None:
            task_instruction += self.demos + '\n'
        history_context = self.decision_context_start.format(n_interactions) + self.represent_history()
        query = self.env.get_query_prompt(state, side_info=None)

        response = self.generate(task_instruction + history_context + query)
        return response

    def update(self, state: State, action: int, reward: float, info: Dict[str, Any]) -> None:
        assert 'interaction' in info
        assert type(info['interaction']) is cb.VerbalInteraction
        self.interaction_history.append(info['interaction'])

    def represent_history(self) -> str:
        return self._represent_interaction_history(self.env.action_names, self.env.bandit_scenario.action_unit,
                                                   self.history_context_len)

    def reset(self):
        super().reset()  # MABAgent.reset()
        self.interaction_history = []

    def get_task_instruction(self) -> str:
        task_instruction = self.env.get_task_instruction()
        return task_instruction

    def get_action_history(self) -> str:
        history_context = self.represent_history()
        return history_context

    def get_decision_query(self, state: State) -> str:
        query = self.env.get_query_prompt(state, side_info=None)
        return query


class OracleLLMMABAgent(LLMMABAgent):
    """Not a full agent"""

    def __init__(self, env: VerbalBandit, oracle_agent: MABAgent,
                 model: str = "gpt-3.5-turbo", history_context_len=1000, verbose=False):
        """
        The oracle_agent will take action
        """

        super().__init__(env, model, history_context_len, verbose)
        self.oracle_agent = oracle_agent
        self.verbose = verbose

    def act(self) -> str:
        action_idx = self.oracle_agent.act()
        return self.env.action_names[action_idx]

    def update(self, action: int, reward: float, info: Dict[str, Any]) -> None:
        # note that we don't have access to the expected reward on the agent side
        assert 'interaction' in info
        assert type(info['interaction']) is mab.VerbalInteraction
        self.interaction_history.append(info['interaction'])
        self.oracle_agent.update(action, reward, info)

    def reset(self):
        super().reset()
        self.oracle_agent.reset()


class OracleLLMCBAgent(LLMCBAgent):
    """Not a full agent"""

    def __init__(self, env: VerbalBandit, oracle_agent: CBAgent,
                 model: str = "gpt-3.5-turbo", history_context_len=1000, verbose=False):
        """
        The oracle_agent will take action
        """

        super().__init__(env, model, history_context_len, verbose)
        self.oracle_agent = oracle_agent
        self.verbose = verbose

    def act(self, state: State) -> str:
        action_idx = self.oracle_agent.act(state)
        return self.env.action_names[action_idx]

    def update(self, state: State, action: int, reward: float, info: Dict[str, Any]) -> None:
        assert 'interaction' in info
        assert type(info['interaction']) is cb.VerbalInteraction
        self.interaction_history.append(info['interaction'])
        self.oracle_agent.update(state, action, reward, info)

    def reset(self):
        super().reset()
        self.oracle_agent.reset()


class LLMMABAgentSH(LLMMABAgent, SummaryHistoryFunc, SampleWithLLMAgent):
    # MAB SH Agent
    ...


class LLMMABAgentRH(LLMMABAgent, MABRawHistoryFunc, SampleWithLLMAgent):
    # MAB RH Agent
    ...


class LLMCBAgentRH(LLMCBAgent, CBRawHistoryFunc, SampleWithLLMAgent):
    # CB RH Agent
    ...


class OracleLLMMABAgentSH(OracleLLMMABAgent, SummaryHistoryFunc, SampleWithLLMAgent):
    ...


class OracleLLMMAbAgentRH(OracleLLMMABAgent, MABRawHistoryFunc, SampleWithLLMAgent):
    ...


class OracleLLMCBAgentRH(OracleLLMCBAgent, CBRawHistoryFunc, SampleWithLLMAgent):
    ...


class MABSummaryHistoryFuncWithAlgorithmGuide(SummaryHistoryFunc):
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


class CBRawHistoryFuncWithAlgorithmGuide(CBRawHistoryFunc):
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

    def _represent_interaction_history(self, action_names: List[str], action_unit: str,
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


class LLMMABAgentSHWithAG(LLMMABAgent, LLM, MABSummaryHistoryFuncWithAlgorithmGuide,
                          SampleWithLLMAgent):
    def __init__(self, env: VerbalBandit,
                 ag: UCBGuide,
                 model: str = "gpt-3.5-turbo",
                 history_context_len=1000,
                 verbose=False):
        LLMMABAgent.__init__(self, env)
        LLM.__init__(self, model)
        MABSummaryHistoryFuncWithAlgorithmGuide.__init__(self, ag,
                                                         history_context_len)
        self.verbose = verbose

    def update(self, action: int, reward: float, info: Dict[str, Any]) -> None:
        # note that we don't have access to the expected reward on the agent side
        super().update(action, reward, info)
        self.update_info_history(self.ag.get_actions_guide_info())
        self.update_algorithm_guide(action, reward, info)

    def reset(self):
        super().reset()  # LLMMABAgent.reset()
        self.ag.agent.reset()


class LLMCBAgentRHWithAG(LLMCBAgent, LLM, CBRawHistoryFuncWithAlgorithmGuide,
                         SampleWithLLMAgent):
    def __init__(self, env: VerbalBandit,
                 ag: LinUCBGuide,
                 model: str = "gpt-3.5-turbo",
                 history_context_len=1000,
                 verbose=False):
        LLMCBAgent.__init__(self, env)
        LLM.__init__(self, model)
        CBRawHistoryFuncWithAlgorithmGuide.__init__(self, ag,
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
        super().reset()  # LLMCBAgent.reset()
        self.ag_info_history = []
        self.ag.agent.reset()

    def act(self, state: State) -> str:
        """Generate next action using LLM and context."""
        # Implement LLM-based contextual action selection
        task_instruction = self.env.get_task_instruction()
        history_context = self.represent_history()

        ag_info = self.ag.get_state_actions_guide_info(state)
        snippet = ""
        for i, action_info in enumerate(ag_info):
            # normal format
            # snippet += '\n' + action_names[i].split(") (")[0] + ")" + ": " + action_info.to_str()

            # JSON-like format used in the paper
            snippet += '\n{\"' + self.env.action_names[i] + ": " + action_info.to_str(
                json_fmt=True) + "}"
        snippet += '\n'

        query = self.decision_context_start + self.env.get_query_prompt(state, side_info=snippet)

        response = self.generate(task_instruction + history_context + query)
        return response


class OracleLLMMABAgentSHWithAG(OracleLLMMABAgent, LLM, MABSummaryHistoryFuncWithAlgorithmGuide,
                                SampleWithLLMAgent):
    def __init__(self, env: VerbalBandit,
                 ag: UCBGuide,
                 oracle_agent: MABAgent,
                 model: str = "gpt-3.5-turbo",
                 history_context_len=1000,
                 verbose=False):
        OracleLLMMABAgent.__init__(self, env, oracle_agent, model, history_context_len, verbose)
        LLM.__init__(self, model)
        MABSummaryHistoryFuncWithAlgorithmGuide.__init__(self, ag,
                                                         history_context_len)

    def update(self, action: int, reward: float, info: Dict[str, Any]) -> None:
        # note that we don't have access to the expected reward on the agent side
        assert 'interaction' in info
        assert type(info['interaction']) is mab.VerbalInteraction
        self.interaction_history.append(info['interaction'])
        self.oracle_agent.update(action, reward, info)
        self.update_info_history(self.ag.get_actions_guide_info())
        self.update_algorithm_guide(action, reward, info)

    def reset(self):
        super().reset()
        self.oracle_agent.reset()
        self.ag.agent.reset()


class OracleLLMCBAgentRHWithAG(OracleLLMCBAgent, LLM, CBRawHistoryFuncWithAlgorithmGuide,
                               SampleWithLLMAgent):
    def __init__(self, env: VerbalBandit,
                 ag: LinUCBGuide,
                 oracle_agent: CBAgent,
                 model: str = "gpt-3.5-turbo",
                 history_context_len=1000,
                 verbose=False):
        OracleLLMCBAgent.__init__(self, env, oracle_agent, model, history_context_len, verbose)
        LLM.__init__(self, model)
        CBRawHistoryFuncWithAlgorithmGuide.__init__(self, ag,
                                                    history_context_len)

    def update(self, state: State, action: int, reward: float, info: Dict[str, Any]) -> None:
        assert 'interaction' in info
        assert type(info['interaction']) is cb.VerbalInteraction
        self.interaction_history.append(info['interaction'])
        self.oracle_agent.update(state, action, reward, info)
        self.update_info_history(self.ag.get_state_actions_guide_info(state))
        self.update_algorithm_guide(state, action, reward, info)

    def reset(self):
        super().reset()
        self.oracle_agent.reset()
        self.ag_info_history = []
        self.ag.agent.reset()

    def get_decision_query(self, state: State) -> str:
        # this is the only thing that challenges with side_info, for the current step
        ag_info = self.ag.get_state_actions_guide_info(state)
        snippet = ""
        for i, action_info in enumerate(ag_info):
            # normal format
            # snippet += '\n' + action_names[i].split(") (")[0] + ")" + ": " + action_info.to_str()

            # JSON-like format used in the paper
            snippet += '\n{\"' + self.env.action_names[i] + ": " + action_info.to_str(
                json_fmt=True) + "}"
        snippet += '\n'

        query = self.env.get_query_prompt(state, side_info=snippet)
        return query


class LLMAgent:
    @classmethod
    def build(cls, env, *args, **kwargs):
        # Extract ag, oracle_agent, and summary from either args or kwargs
        # we disallow passing in summary flag as an argument (because we can only determine `bool` from `args`, which is too risky)
        ag = None
        oracle_agent = None

        remaining_args = []
        for arg in args:
            if hasattr(arg, 'get_action_guide_info'):  # Check if arg is algorithm guide
                ag = arg
            elif hasattr(arg, 'act') and hasattr(arg,
                                                 'update'):  # Check if arg is oracle agent; can be LLM or a classic agent
                oracle_agent = arg
            # elif isinstance(arg, bool):  # Check if arg is summary flag
            #     summary = arg
            else:
                remaining_args.append(arg)

        # Also check kwargs
        ag = ag or kwargs.pop('ag', None)
        oracle_agent = oracle_agent or kwargs.pop('oracle_agent', None)
        summary = kwargs.pop('summary', False)

        # Determine if environment is contextual bandit or multi-armed bandit
        if hasattr(env, 'action_names'):
            is_cb = hasattr(env.core_bandit, 'sample_state')
        else:
            is_cb = hasattr(env, 'sample_state')

        if oracle_agent:
            if is_cb:
                if ag:
                    # When AG is present, always use SH
                    return OracleLLMCBAgentRHWithAG(env, ag, oracle_agent, *remaining_args, **kwargs)
                return OracleLLMCBAgentRH(env, oracle_agent, *remaining_args, **kwargs)
            else:  # MAB
                if ag:
                    # When AG is present, always use SH
                    return OracleLLMMABAgentSHWithAG(env, ag, oracle_agent, *remaining_args, **kwargs)
                # For MAB without AG, respect summary flag
                if summary:
                    return OracleLLMMABAgentSH(env, oracle_agent, *remaining_args, **kwargs)
                return OracleLLMMAbAgentRH(env, oracle_agent, *remaining_args, **kwargs)
        else:
            if is_cb:
                if ag:
                    # When AG is present, always use SH
                    return LLMCBAgentRHWithAG(env, ag, *remaining_args, **kwargs)
                return LLMCBAgentRH(env, *remaining_args, **kwargs)
            else:  # MAB
                if ag:
                    # When AG is present, always use SH
                    return LLMMABAgentSHWithAG(env, ag, *remaining_args, **kwargs)
                # For MAB without AG, respect summary flag
                if summary:
                    return LLMMABAgentSH(env, *remaining_args, **kwargs)
                return LLMMABAgentRH(env, *remaining_args, **kwargs)
