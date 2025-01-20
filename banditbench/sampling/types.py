import json
import numpy as np
from pathlib import Path
from typing import Union, Dict, Any, List
from banditbench.tasks.env import VerbalBandit, Bandit
from banditbench.tasks.cb.env import ContextualBandit
from banditbench.tasks.types import Trajectory
from banditbench.agents.types import Agent, ActionInfo

from banditbench.utils import plot_cumulative_reward, plot_multi_cumulative_reward

"""
DatasetBuffer has 3 components:
 - Trajectories: List of Trajectory objects (bare minimum) (all agents will have this) (this is just the raw interaction history with the environment)
 - ActionInfos: Additional information at each step of the decision (some agents have them, some don't) (for agent that has them, this is not exposed currently)
 - VerbalPrompts: The prompt, task description that was sent into LLM to get the label (For LLM agent, and oracleLLM agent) (these are also not exposed)
"""


class Data(dict):
    # this is on the trajectory level -- a single trajectory
    trajectory: Trajectory
    ag_info: Union[List[List[ActionInfo]], None]
    verbal_prompts: Union[List[Dict[str, str]], None]

    def __init__(self, trajectory: Trajectory, action_info: Union[List[List[ActionInfo]], None] = None,
                 verbal_prompts: Union[List[Dict[str, str]], None] = None):
        super().__init__()
        self['trajectory'] = trajectory
        self['action_info'] = action_info
        self['verbal_prompts'] = verbal_prompts

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(f"'Data' object has no attribute '{name}'")

    def __setattr__(self, name, value):
        self[name] = value

    def __iter__(self):
        return iter([self.trajectory, self.action_info, self.verbal_prompts])


class DatasetBuffer:
    # this is on the dataset level -- a dataset of trajectories
    def __init__(self, trajectories=None, ag_info=None, verbal_prompts=None):
        self.trajectories = trajectories or []
        self.ag_info = ag_info or []
        self.verbal_prompts = verbal_prompts or []

    def append(self, trajectory: Trajectory, action_info: Union[List[List[ActionInfo]], None] = None,
               verbal_prompt: Union[List[Dict[str, str]], None] = None):
        self.trajectories.append(trajectory)
        if action_info is not None:
            self.ag_info.append(action_info)
        if verbal_prompt is not None:
            self.verbal_prompts.append(verbal_prompt)

    def add(self, trajectory: Trajectory, action_info: Union[List[List[ActionInfo]], None] = None,
            verbal_prompt: Union[List[Dict[str, str]], None] = None):
        self.append(trajectory, action_info, verbal_prompt)

    def clear(self):
        self.trajectories.clear()
        self.ag_info.clear()
        self.verbal_prompts.clear()

    def __len__(self):
        return len(self.trajectories)

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            # Handle slice indexing
            trajectories = self.trajectories[idx]
            ag_info = self.ag_info[idx] if self.ag_info else None
            verbal_prompts = self.verbal_prompts[idx] if self.verbal_prompts else None

            # Create new buffer with sliced data
            new_buffer = DatasetBuffer(trajectories, ag_info, verbal_prompts)
            return new_buffer
        else:
            # Handle single index
            return Data(
                trajectory=self.trajectories[idx],
                action_info=self.ag_info[idx] if self.ag_info else None,
                verbal_prompts=self.verbal_prompts[idx] if self.verbal_prompts else None
            )

    def __str__(self):
        return f"DatasetBuffer({len(self)} trajectories)"

    def __repr__(self):
        return str(self)

    def __iter__(self):
        for i in range(len(self)):
            yield Data(
                trajectory=self.trajectories[i],
                action_info=self.ag_info[i] if self.ag_info else None,
                verbal_prompts=self.verbal_prompts[i] if self.verbal_prompts else None
            )

    def __add__(self, other):
        if isinstance(other, DatasetBuffer):
            result = DatasetBuffer()
            result.trajectories.extend(self.trajectories)
            result.trajectories.extend(other.trajectories)
            if self.ag_info and other.ag_info:
                result.ag_info.extend(self.ag_info)
                result.ag_info.extend(other.ag_info)
            if self.verbal_prompts and other.verbal_prompts:
                result.verbal_prompts.extend(self.verbal_prompts)
                result.verbal_prompts.extend(other.verbal_prompts)
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
                traj.model_dump() for traj in self.trajectories
            ]
        }

        if self.ag_info:
            data['ag_info'] = [
                [[info.model_dump() for info in action_infos]
                 for action_infos in interaction_infos]
                for interaction_infos in self.ag_info
            ]

        if self.verbal_prompts:
            data['verbal_prompts'] = self.verbal_prompts

        with open(filepath, 'w') as f:
            json.dump(data, f)

    @classmethod
    def load(cls, filepath: str) -> 'DatasetBuffer':
        """Load a dataset buffer from a JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f)

        trajectories = [Trajectory.model_validate(traj_data) for traj_data in data['trajectories']]
        buffer = cls(trajectories=trajectories)

        if 'ag_info' in data and data['ag_info']:
            buffer.action_infos = [
                [
                    [ActionInfo.model_validate(info) for info in action_infos]
                    for action_infos in interaction_infos
                ]
                for interaction_infos in data['ag_info']
            ]

        if 'verbal_prompts' in data:
            buffer.verbal_prompts = data['verbal_prompts']

        return buffer

    def save(self, file):
        self.dump(file)

    def plot_performance(self, *args, title=None, labels=None, filename=None):
        # plot the mean performance over all trajectories stored in the dataset
        config_name_to_all_rewards = {}
        
        # Parse args to find title if provided
        buffers = []
        for arg in args:
            if isinstance(arg, str):
                if arg.endswith('.png') or arg.endswith('.jpg') or arg.endswith('.jpeg') or arg.endswith('.pdf'):
                    filename = arg
                else:
                    title = arg
            else:
                buffers.append(arg)
        
        # Get rewards from current buffer
        current_rewards = []
        for trajectory in self.trajectories:
            rewards = []
            for interaction in trajectory:
                rewards.append(interaction.reward)
            current_rewards.append(rewards)
            
        config_name = labels[0] if labels and len(labels) > 0 else 'Agent 1'
        config_name_to_all_rewards[config_name] = current_rewards
        
        # Get rewards from other buffers if provided
        if buffers:
            for i, buffer in enumerate(buffers):
                buffer_rewards = []
                for trajectory in buffer.trajectories:
                    rewards = []
                    for interaction in trajectory:
                        rewards.append(interaction.reward)
                    buffer_rewards.append(rewards)
                    
                buffer_name = labels[i+1] if labels and len(labels) > i+1 else f'Agent {i+2}'
                config_name_to_all_rewards[buffer_name] = buffer_rewards
            
        horizon = len(current_rewards[0])
        plot_multi_cumulative_reward(config_name_to_all_rewards, horizon, title, filename)

    def to_sft_data(self, file=None):
        """
        'task_instruction': task_instruction,
        'action_history': action_history,
        'decision_query': decision_query,
        'label': action_verbal
        """
        # [{}]
        data = []
        for trajectory_prompts in self.verbal_prompts:
            traj_prompt = []
            for i, trajectory_step_prompt in enumerate(trajectory_prompts):
                traj_prompt.append({'step': i,
                                    "prompt": trajectory_step_prompt['task_instruction'] + trajectory_step_prompt[
                                        'action_history'] + trajectory_step_prompt['decision_query'],
                                    "label": trajectory_step_prompt['label']})
            data.append(traj_prompt)

        if file:
            if isinstance(file, str):
                filepath = file
            else:
                filepath = file.name

            with open(filepath, 'w') as f:
                json.dump(data, f)
        else:
            return data

    def save_sft_data(self, file=None):
        return self.to_sft_data(file)
