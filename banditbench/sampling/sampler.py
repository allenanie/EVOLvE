"""
Hosts sampler mixin (used by agent to add create dataset functionality)
"""
import json
import numpy as np
from pathlib import Path
from typing import Union, Dict, Any
from banditbench.tasks.typing import Trajectory


class DatasetBuffer(list):
    def __init__(self):
        super().__init__()

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


class DataCollection:
    def collect(self, env, buffer, n_steps=100000):
        pass
