"""
Hosts sampler mixin (used by agent to add create dataset functionality)
"""
import json
import numpy as np
from pathlib import Path
from typing import Union, Dict, Any, List
from banditbench.tasks.typing import Trajectory
from banditbench.agents.typing import Agent, ActionInfo

from banditbench.utils import plot_cumulative_reward

"""
DatasetBuffer has 3 components:
 - Trajectories: List of Trajectory objects (bare minimum) (all agents will have this) (this is just the raw interaction history with the environment)
 - ActionInfos: Additional information at each step of the decision (some agents have them, some don't) (for agent that has them, this is not exposed currently)
 - VerbalPrompts: The prompt, task description that was sent into LLM to get the label (For LLM agent, and oracleLLM agent) (these are also not exposed)
"""

class DatasetBuffer(list):
    def __init__(self, trajectories=None, action_infos=None):
        super().__init__(trajectories or [])
        self.action_infos = action_infos or []
        self.verbal_prompts = []

    def add(self, trajectory: Trajectory, action_info: Union[List[List[ActionInfo]], None] = None):
        # action_info: [Traject_length, Num_Action]
        self.append(trajectory)
        if action_info is not None:
            self.action_infos.append(action_info)

    def clear(self):
        super().clear()
        self.action_infos.clear()

    def __str__(self):
        return f"DatasetBuffer({len(self)} trajectories)"

    def __repr__(self):
        return str(self)

    def __add__(self, other):
        if isinstance(other, DatasetBuffer):
            result = DatasetBuffer()
            result.extend(self)
            result.extend(other)
            result.action_infos.extend(self.action_infos)
            result.action_infos.extend(other.action_infos)
            return result
        else:
            raise ValueError(f"Unsupported type: {type(other)}")

    def dump(self, file):
        """Save the dataset buffer to a JSON file."""
        if isinstance(file, str):
            filepath = file
        else:
            filepath = file.name

        data = {
            'n_trajectories': len(self),
            'trajectories': [
                traj.model_dump() for traj in self
            ],
            'action_infos': [
                [[info.model_dump() for info in action_infos] 
                 for action_infos in interaction_infos]
                for interaction_infos in self.action_infos
            ] if self.action_infos else []
        }

        with open(filepath, 'w') as f:
            json.dump(data, f)

    @classmethod
    def load(cls, filepath: str) -> 'DatasetBuffer':
        """Load a dataset buffer from a JSON file."""
        buffer = cls()

        with open(filepath, 'r') as f:
            data = json.load(f)

        for traj_data in data['trajectories']:
            traj = Trajectory.model_validate(traj_data)
            buffer.append(traj)

        if 'action_infos' in data and data['action_infos']:
            for interaction_infos in data['action_infos']:
                buffer.action_infos.append([
                    [ActionInfo.model_validate(info) for info in action_infos]
                    for action_infos in interaction_infos
                ])

        return buffer

    def save(self, file):
        self.dump(file)

    def to_sft_format(self):
        # The SFT format is a lossy format that only stores strings
        # It's a list of dictionary  [{'task_description': "", 'action_history': "", 'label': ""}]
        # where `label` is the action taken by the agent
        # If there is side ag information, it would be present in the prompt
        # prompt includes history of interations
        pass

    def save_sft_format(self):
        pass

    def plot_performance(self, title=None):
        # plot the mean performance over all trajectories stored in the dataset
        all_rewards = []
        for trajectory in self:
            rewards = []
            for interaction in trajectory:
                rewards.append(interaction.reward)
            all_rewards.append(rewards)
        horizon = len(all_rewards[0])
        plot_cumulative_reward(all_rewards, horizon, title)


class DataCollect:

    def collect(self, env, n_trajectories=1000) -> DatasetBuffer:
        """Collect interactions from environment and store in buffer.
        
        Args:
            env: The environment to collect from (Verbal or non-verbal)
            agent: Agent to collect data with
            n_trajectories: Number of self-improving trajectories to collect
        """
        # Check if environment is verbal by looking for verbal_info property
        is_verbal = hasattr(env, 'action_names')
        is_contextual = hasattr(env, 'feature_dim')

        buffer = DatasetBuffer()

        trajectories_collected = 0
        while trajectories_collected < n_trajectories:
            trajectory = []
            self.reset()

            if is_contextual:
                # Contextual bandit case
                state, _ = env.reset()
                done = False
                while not done:
                    action = self.act(state)
                    new_state, reward, done, info = env.step(state, action)
                    trajectory.append(info['interaction'])
                    self.update(state, action, reward, info)
                    state = new_state
            else:
                # Multi-armed bandit case
                env.reset()
                done = False
                while not done:
                    action = self.act()
                    _, reward, done, info = env.step(action)
                    trajectory.append(info['interaction'])
                    self.update(action, reward, info)

            buffer.append(Trajectory(trajectory))
            trajectories_collected += 1

        return buffer


class DataCollectWithAGInfo:
    # this is the mixin for VerbalGuideAgent

    def collect(self, env, n_trajectories=1000) -> DatasetBuffer:
        # AG has an underlying agent
        # but also provides utility class to load in action info
        # we need to both get the interaction from the underlying agent
        # and collect the action info from the AG
        is_contextual = hasattr(env, 'feature_dim')

        buffer = DatasetBuffer()

        trajectories_collected = 0
        while trajectories_collected < n_trajectories:
            trajectory = []
            ag_info = []

            self.agent.reset()

            if is_contextual:
                # Contextual bandit case
                state, _ = env.reset()
                done = False
                while not done:
                    action = self.agent.act(state)
                    new_state, reward, done, info = env.step(state, action)
                    action_info = self.get_state_actions_guide_info(state)

                    trajectory.append(info['interaction'])
                    ag_info.append(action_info)

                    self.agent.update(state, action, reward, info)
                    state = new_state
            else:
                # Multi-armed bandit case
                env.reset()
                done = False
                while not done:
                    action = self.agent.act()
                    _, reward, done, info = env.step(action)
                    action_info = self.get_actions_guide_info()

                    trajectory.append(info['interaction'])
                    ag_info.append(action_info)

                    self.agent.update(action, reward, info)

            buffer.add(Trajectory(trajectory), ag_info)
            trajectories_collected += 1

        return buffer
