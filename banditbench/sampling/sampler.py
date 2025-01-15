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


class DatasetBuffer:
    def __init__(self, trajectories=None, action_infos=None, verbal_prompts=None):
        self.trajectories = trajectories or []
        self.action_infos = action_infos or []
        self.verbal_prompts = verbal_prompts or []

    def append(self, trajectory: Trajectory, action_info: Union[List[List[ActionInfo]], None] = None,
               verbal_prompt: Union[str, None] = None):
        self.trajectories.append(trajectory)
        if action_info is not None:
            self.action_infos.append(action_info)
        if verbal_prompt is not None:
            self.verbal_prompts.append(verbal_prompt)

    def add(self, trajectory: Trajectory, action_info: Union[List[List[ActionInfo]], None] = None,
            verbal_prompt: Union[str, None] = None):
        self.append(trajectory, action_info, verbal_prompt)

    def clear(self):
        self.trajectories.clear()
        self.action_infos.clear()
        self.verbal_prompts.clear()

    def __len__(self):
        return len(self.trajectories)

    def __getitem__(self, idx):
        return self.trajectories[idx]

    def __str__(self):
        return f"DatasetBuffer({len(self)} trajectories)"

    def __repr__(self):
        return str(self)

    def __add__(self, other):
        if isinstance(other, DatasetBuffer):
            result = DatasetBuffer()
            result.trajectories.extend(self.trajectories)
            result.trajectories.extend(other.trajectories)
            if self.action_infos and other.action_infos:
                result.action_infos.extend(self.action_infos)
                result.action_infos.extend(other.action_infos)
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

        if self.action_infos:
            data['action_infos'] = [
                [[info.model_dump() for info in action_infos]
                 for action_infos in interaction_infos]
                for interaction_infos in self.action_infos
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

        if 'action_infos' in data and data['action_infos']:
            buffer.action_infos = [
                [
                    [ActionInfo.model_validate(info) for info in action_infos]
                    for action_infos in interaction_infos
                ]
                for interaction_infos in data['action_infos']
            ]

        if 'verbal_prompts' in data:
            buffer.verbal_prompts = data['verbal_prompts']

        return buffer

    def save(self, file):
        self.dump(file)

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


class DataCollectWithLLMAgent:
    """This is a mixin for LLMAgent. LLM agent exposes an API for prompts. Note we store the mapped_action from the environment,
    not the direct output from the model."""

    def collect(self, env, n_trajectories=1000) -> DatasetBuffer:
        pass
