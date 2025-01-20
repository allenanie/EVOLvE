from typing import Optional, List, Any, Dict, Union
from banditbench.sampling.sampler import DatasetBuffer
from banditbench.agents.llm import LLMMABAgentBase, LLMCBAgentBase

class FewShot:
    """
    Takes in agent, and returns a FewShot agent
    FewShot.empower(agent, *configs) # becomes a FewShotAgent

    Agent stores the state, not FewShot class

    This is a passthrough class.
    It augments the original agent's behavior with few-shot examples.

    We use class variables to help applying the same config to different agents without reprocessing
    """

    data_buffer: Optional[DatasetBuffer] = None
    filename: Optional[str] = None
    num_examples: int = 5
    skip_first: int = 2
    sample_freq: int = 2
    traj_idx: int = 0  # which trajectory we choose from the datasource

    @classmethod
    def empower(cls, agent: Union[LLMMABAgentBase, LLMCBAgentBase],
                datasource: Optional[Union[DatasetBuffer, str]] = None,
                num_examples: Optional[int] = None,
                skip_first: Optional[int] = None,
                sample_freq: Optional[int] = None,
                traj_idx: Optional[int] = None) -> Union[LLMMABAgentBase, LLMCBAgentBase]:
        """
        This is the general "compilation" function that takes in the agent

        :param skip_first: skip the first few examples (because the decision might not be very complex)
        :param num_examples: The total number of examples in context
        :param sample_freq: For each trajectory, the number of improvement steps are between each example
        """
        if datasource is not None:
            print("received datasource: ", datasource)
            if type(datasource) is str:
                cls.filename = datasource
                cls.data_buffer = DatasetBuffer.load(cls.filename)
            elif type(datasource) is DatasetBuffer:
                cls.data_buffer = datasource
            else:
                raise ValueError("Invalid datasource type")

        if traj_idx is not None:
            cls.traj_idx = traj_idx

        if num_examples is not None:
            cls.num_examples = num_examples

        if skip_first is not None:
            cls.skip_first = skip_first

        if sample_freq is not None:
            cls.sample_freq = sample_freq

        # then determine MAB or CB to load the examples into agent
        if isinstance(agent, LLMMABAgentBase):
            agent.demos = cls.load_few_shot_mab_examples(agent)
        elif isinstance(agent, LLMCBAgentBase):
            agent.demos = cls.load_few_shot_cb_examples(agent)

        return agent

    @classmethod
    def load_few_shot_mab_examples(cls, agent: Union[LLMMABAgentBase, LLMCBAgentBase]) -> Optional[str]:
        """This has to be different """
        if cls.data_buffer is None:
            return None
        else:
            fewshot_prompt = "\n" + agent.env.bandit_scenario.fewshot_prompt
            fewshot_prompt += "========================"
            start_idx = cls.skip_first
            # first trajectory in the data buffer
            examples = cls.data_buffer[cls.traj_idx].verbal_prompts[start_idx::cls.sample_freq][:cls.num_examples]
            if len(examples) == 0:
                return None
            for example in examples:
                fewshot_prompt += example['action_history']
                # query
                fewshot_prompt += example['decision_query']
                fewshot_prompt += f"\n{example['label']}\n"
                fewshot_prompt += "========================"

            return fewshot_prompt

    @classmethod
    def load_few_shot_cb_examples(cls, agent: Union[LLMMABAgentBase, LLMCBAgentBase]) -> Optional[str]:
        """This has to be different """
        if cls.data_buffer is None:
            return None
        else:
            fewshot_prompt = "\n" + agent.env.bandit_scenario.fewshot_prompt
            fewshot_prompt += "============Example Trajectory============"
            start_idx = cls.skip_first
            examples = cls.data_buffer[cls.traj_idx].verbal_prompts[start_idx::cls.sample_freq][:cls.num_examples]
            if len(examples) == 0:
                return None
            for example in examples:
                fewshot_prompt += example['action_history']
                # query
                fewshot_prompt += example['decision_query']
                fewshot_prompt += f"\n{example['label']}\n"
                fewshot_prompt += "=============Example Trajectory==========="

            return fewshot_prompt
