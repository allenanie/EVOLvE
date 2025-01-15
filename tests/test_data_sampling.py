import tempfile
import os
import numpy as np
import pytest

from banditbench.sampling.sampler import DatasetBuffer, Trajectory
from banditbench.tasks.mab import BernoulliBandit, VerbalMultiArmedBandit

from banditbench.agents.classics import UCBAgent
from banditbench.agents.guides import UCBGuide

from banditbench.tasks.mab.env import Interaction as MABInteraction, VerbalInteraction as MABVerbalInteraction
from banditbench.tasks.cb.env import Interaction as CBInteraction, VerbalInteraction as CBVerbalInteraction


@pytest.fixture
def temp_files():
    files = []
    yield files
    # Cleanup after tests
    for file in files:
        if os.path.exists(file):
            os.remove(file)


def test_save_and_load(temp_files):
    core_bandit = BernoulliBandit(2, 10, [0.5, 0.2], 123)
    verbal_bandit = VerbalMultiArmedBandit(core_bandit, "VideoWatching")

    # Create a temporary file
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.json')
    temp_files.append(temp_file.name)
    temp_file.close()

    # Create test data
    buffer = DatasetBuffer()

    _, reward, done, info = verbal_bandit.step('A')
    inter1 = info['interaction']

    _, reward, done, info = verbal_bandit.step('B')
    inter2 = info['interaction']

    test_traj = inter1 + inter2

    buffer.append(test_traj)

    # Test save
    buffer.dump(temp_file.name)
    assert os.path.exists(temp_file.name)

    # Test load
    loaded_buffer = DatasetBuffer.load(temp_file.name)
    assert len(loaded_buffer) == 1

    loaded_traj = loaded_buffer[0]
    assert len(loaded_traj) == 2
    assert loaded_traj[0].raw_action == inter1.raw_action
    assert loaded_traj[0].reward == inter1.reward


def test_multiple_trajectories(temp_files):
    core_bandit = BernoulliBandit(2, 10, [0.5, 0.2], 123)
    verbal_bandit = VerbalMultiArmedBandit(core_bandit, "VideoWatching")

    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.json')
    temp_files.append(temp_file.name)
    temp_file.close()

    buffer = DatasetBuffer()
    n_trajectories = 5

    # Add multiple trajectories
    for _ in range(n_trajectories):
        _, reward, done, info = verbal_bandit.step('A')
        inter1 = info['interaction']
        _, reward, done, info = verbal_bandit.step('B')
        inter2 = info['interaction']

        traj = inter1 + inter2
        buffer.append(traj)

    # Save and load
    buffer.save(temp_file.name)
    loaded_buffer = DatasetBuffer.load(temp_file.name)

    # Verify
    assert len(loaded_buffer) == n_trajectories
    for traj in loaded_buffer:
        assert len(traj) == 2
        assert isinstance(traj[0], MABVerbalInteraction)
        assert isinstance(traj[1], MABVerbalInteraction)

def test_databuffer_slicing():
    core_bandit = BernoulliBandit(2, 10, [0.5, 0.2], 123)
    verbal_bandit = VerbalMultiArmedBandit(core_bandit, "VideoWatching")

    buffer = DatasetBuffer()
    n_trajectories = 3

    # Add multiple trajectories
    for _ in range(n_trajectories):
        _, reward, done, info = verbal_bandit.step('A')
        inter1 = info['interaction']
        _, reward, done, info = verbal_bandit.step('B')
        inter2 = info['interaction']
        _, reward, done, info = verbal_bandit.step('B')
        inter3 = info['interaction']

        traj = inter1 + inter2 + inter3
        buffer.append(traj)

    # Test slicing
    sliced_buffer = buffer[:1]
    assert len(sliced_buffer) == 1
    assert sliced_buffer[0] == buffer[0]
    sliced_buffer = buffer[1:2]
    assert sliced_buffer[0] == buffer[1]
    assert len(sliced_buffer) == 1
    sliced_buffer = buffer[0:2]
    assert len(sliced_buffer) == 2
    assert sliced_buffer[0] == buffer[0]
    assert sliced_buffer[1] == buffer[2]

def test_mab_sampling():
    core_bandit = BernoulliBandit(2, 200, [0.2, 0.5], 123)

    agent = UCBAgent(core_bandit)
    dataset = agent.collect(core_bandit, 100)
    assert len(dataset) == 100
    dataset.plot_performance()

def test_mab_ag_sampling(temp_files):
    core_bandit = BernoulliBandit(2, 200, [0.2, 0.5], 123)

    guide = UCBGuide(UCBAgent(core_bandit))
    dataset = guide.collect(core_bandit, 20)
    assert len(dataset) == 20

    assert len(dataset.action_infos) == 20

    # the 1st trajectory, 1st interaction, 1st action's info
    print(dataset.action_infos[0][0][0].to_str())

    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.json')
    temp_files.append(temp_file.name)
    temp_file.close()

    dataset.save(temp_file.name)
    loaded_buffer = DatasetBuffer.load(temp_file.name)

    assert len(loaded_buffer) == 20
