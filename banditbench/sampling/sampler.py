"""
Hosts sampler mixin (used by agent to add create dataset functionality)
"""
import json
import numpy as np
from pathlib import Path
from typing import Union, Dict, Any
from banditbench.tasks.typing import Trajectory
from banditbench.agents.typing import Agent

from banditbench.utils import plot_cumulative_reward


class DatasetBuffer(list):
    def __init__(self, trajectories=None):
        super().__init__(trajectories or [])

    def add(self, trajectory: Trajectory):
        self.append(trajectory)

    def clear(self):
        super().clear()

    def __str__(self):
        return f"DatasetBuffer({len(self)} trajectories)"

    def __repr__(self):
        return str(self)

    def __add__(self, other):
        if isinstance(other, DatasetBuffer):
            result = DatasetBuffer()
            result.extend(self)
            result.extend(other)
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
            ]
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


class DataCollection:

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
