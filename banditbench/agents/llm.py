from typing import Optional, List, Any, Dict

import litellm

from banditbench.agents.context import ModelContextLayer, RawContextLayerMAB, RawContextLayerCB, SummaryContextLayerMAB, \
    SummaryAlgoGuideLayerMAB, RawContextAlgoGuideLayerCB

from banditbench.agents.guides import UCBGuide, LinUCBGuide
from banditbench.tasks.types import State
from banditbench.tasks.env import VerbalBandit
from banditbench.sampling.sampler import AgentSampleBase

import banditbench.tasks.cb as cb
import banditbench.tasks.mab as mab

from banditbench.agents.types import MABAgent, CBAgent


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


class LLMMABAgentBase(MABAgent, LLM, ModelContextLayer, AgentSampleBase):
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
        ModelContextLayer.__init__(self, history_context_len)
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
        return self.represent_interaction_context(self.env.action_names, self.env.bandit_scenario.action_unit,
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


class LLMCBAgentBase(CBAgent, LLM, ModelContextLayer, AgentSampleBase):
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
        ModelContextLayer.__init__(self, history_context_len)
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

    def represent_history(self) -> str:
        return self.represent_interaction_context(self.env.action_names, self.env.bandit_scenario.action_unit,
                                                  self.history_context_len)

    def update(self, state: State, action: int, reward: float, info: Dict[str, Any]) -> None:
        assert 'interaction' in info
        assert type(info['interaction']) is cb.VerbalInteraction
        self.interaction_history.append(info['interaction'])

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


class OracleLLMMABAgent(LLMMABAgentBase):
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


class OracleLLMCBAgent(LLMCBAgentBase):
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


class LLMMABAgentSH(LLMMABAgentBase, SummaryContextLayerMAB):
    # MAB SH Agent
    name = "MAB_SH_Agent"


class LLMMABAgentRH(LLMMABAgentBase, RawContextLayerMAB):
    # MAB RH Agent
    name = "MAB_RH_Agent"


class LLMCBAgentRH(LLMCBAgentBase, RawContextLayerCB):
    # CB RH Agent
    name = "CB_RH_Agent"


class OracleLLMMABAgentSH(OracleLLMMABAgent, SummaryContextLayerMAB):
    name = "Oracle_MAB_SH_Agent"


class OracleLLMMAbAgentRH(OracleLLMMABAgent, RawContextLayerMAB):
    name = "Oracle_MAB_RH_Agent"


class OracleLLMCBAgentRH(OracleLLMCBAgent, RawContextLayerCB):
    name = "Oracle_CB_RH_Agent"


class LLMMABAgentSHWithAG(LLMMABAgentBase, LLM, SummaryAlgoGuideLayerMAB):
    name = "MAB_SH_AG_Agent"

    def __init__(self, env: VerbalBandit,
                 ag: UCBGuide,
                 model: str = "gpt-3.5-turbo",
                 history_context_len=1000,
                 verbose=False):
        LLMMABAgentBase.__init__(self, env)
        LLM.__init__(self, model)
        SummaryAlgoGuideLayerMAB.__init__(self, ag,
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


class LLMCBAgentRHWithAG(LLMCBAgentBase, LLM, RawContextAlgoGuideLayerCB):
    def __init__(self, env: VerbalBandit,
                 ag: LinUCBGuide,
                 model: str = "gpt-3.5-turbo",
                 history_context_len=1000,
                 verbose=False):
        LLMCBAgentBase.__init__(self, env)
        LLM.__init__(self, model)
        RawContextAlgoGuideLayerCB.__init__(self, ag,
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
        history_context = self.represent_interaction_context()

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


class OracleLLMMABAgentSHWithAG(OracleLLMMABAgent, LLM, SummaryAlgoGuideLayerMAB):
    name = "Oracle_MAB_SH_AG_Agent"

    def __init__(self, env: VerbalBandit,
                 ag: UCBGuide,
                 oracle_agent: MABAgent,
                 model: str = "gpt-3.5-turbo",
                 history_context_len=1000,
                 verbose=False):
        OracleLLMMABAgent.__init__(self, env, oracle_agent, model, history_context_len, verbose)
        LLM.__init__(self, model)
        SummaryAlgoGuideLayerMAB.__init__(self, ag,
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


class OracleLLMCBAgentRHWithAG(OracleLLMCBAgent, LLM, RawContextAlgoGuideLayerCB):
    name = "Oracle_CB_RH_AG_Agent"

    def __init__(self, env: VerbalBandit,
                 ag: LinUCBGuide,
                 oracle_agent: CBAgent,
                 model: str = "gpt-3.5-turbo",
                 history_context_len=1000,
                 verbose=False):
        OracleLLMCBAgent.__init__(self, env, oracle_agent, model, history_context_len, verbose)
        LLM.__init__(self, model)
        RawContextAlgoGuideLayerCB.__init__(self, ag,
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


class LLMAgentBuilder:
    """This is only a partially initialized agent"""

    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    def build_with_env(self, env):
        return LLMAgent.build_with_env(env, *self.args, **self.kwargs)

    def build(self, env):
        return LLMAgent.build_with_env(env, *self.args, **self.kwargs)


class LLMAgent:
    @classmethod
    def build_with_env(cls, env, *args, **kwargs):
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

    @classmethod
    def build(cls, *args, **kwargs):
        # Extract ag, oracle_agent, and summary from either args or kwargs
        # we disallow passing in summary flag as an argument (because we can only determine `bool` from `args`, which is too risky)

        ag = None
        oracle_agent = None
        env = None

        remaining_args = []
        for arg in args:
            if hasattr(arg, 'get_action_guide_info'):  # Check if arg is algorithm guide
                ag = arg
            elif hasattr(arg, 'act') and hasattr(arg,
                                                 'update'):  # Check if arg is oracle agent; can be LLM or a classic agent
                oracle_agent = arg
            elif hasattr(args, 'action_names'):
                env = arg
            else:
                remaining_args.append(arg)

        # Also check kwargs
        ag = ag or kwargs.pop('ag', None)
        oracle_agent = oracle_agent or kwargs.pop('oracle_agent', None)
        summary = kwargs.pop('summary', False)
        env = kwargs.pop('env', None)

        if env is not None:
            return cls.build_with_env(env, ag, oracle_agent, summary, *remaining_args, **kwargs)
        else:
            return LLMAgentBuilder(ag, oracle_agent, summary=summary)
