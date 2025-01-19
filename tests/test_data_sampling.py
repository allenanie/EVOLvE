import tempfile
import os
import numpy as np
import pytest

from banditbench.sampling.sampler import DatasetBuffer, Trajectory
from banditbench.tasks.mab import BernoulliBandit, VerbalMultiArmedBandit

from banditbench.agents.classics import UCBAgent, LinUCBAgent
from banditbench.agents.guides import UCBGuide, LinUCBGuide
from banditbench.agents.llm import (LLMMABAgentRH, LLMMABAgentSH, LLMCBAgentRH, LLMCBAgentRHWithAG, LLMMABAgentSHWithAG,
                                    OracleLLMCBAgentRH, OracleLLMCBAgentRHWithAG, OracleLLMMABAgentSH,
                                    OracleLLMMAbAgentRH, OracleLLMMABAgentSHWithAG)


from banditbench.tasks.mab.env import Interaction as MABInteraction, VerbalInteraction as MABVerbalInteraction
from banditbench.tasks.cb.env import Interaction as CBInteraction, VerbalInteraction as CBVerbalInteraction

from banditbench.tasks.cb.movielens import MovieLens, MovieLensVerbal


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


def test_mab_sampling(temp_files):
    core_bandit = BernoulliBandit(2, 200, [0.2, 0.5], 123)

    agent = UCBAgent(core_bandit)
    dataset = agent.in_context_learn(core_bandit, 100)
    assert len(dataset) == 100
    dataset.plot_performance()

    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.json')
    temp_files.append(temp_file.name)
    temp_file.close()

    dataset.save(temp_file.name)
    loaded_buffer = DatasetBuffer.load(temp_file.name)

    assert len(loaded_buffer) == 100


def test_mab_ag_sampling(temp_files):
    core_bandit = BernoulliBandit(2, 200, [0.2, 0.5], 123)

    guide = UCBGuide(UCBAgent(core_bandit))
    dataset = guide.in_context_learn(core_bandit, 20)
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


env = None
verbal_env = None


def init_cb_env():
    global env
    global verbal_env

    if env is None:
        env = MovieLens('100k-ratings', num_arms=5, horizon=10, rank_k=5, mode='train',
                        save_data_dir='./tensorflow_datasets/')
    if verbal_env is None:
        verbal_env = MovieLensVerbal(env)


def test_mab_llm_rh_sampling(temp_files):
    # Setup environment
    core_bandit = BernoulliBandit(2, 10, [0.2, 0.5], 123)
    verbal_bandit = VerbalMultiArmedBandit(core_bandit, "VideoWatching")

    # Test LLMMABAgentRH
    agent = LLMMABAgentRH(verbal_bandit, "gpt-3.5-turbo", history_context_len=1000)
    buffer = agent.in_context_learn(verbal_bandit, n_trajs=2)
    assert len(buffer) == 2
    assert len(buffer[0].verbal_prompts) == 10

    # Test save/load
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.json')
    temp_files.append(temp_file.name)
    temp_file.close()

    buffer.dump(temp_file.name)
    loaded_buffer = DatasetBuffer.load(temp_file.name)
    assert len(loaded_buffer) == 2


def test_mab_llm_sh_sampling(temp_files):
    # Setup environment
    core_bandit = BernoulliBandit(2, 10, [0.2, 0.5], 123)
    verbal_bandit = VerbalMultiArmedBandit(core_bandit, "VideoWatching")

    # Test LLMMABAgentSH
    agent = LLMMABAgentSH(verbal_bandit, "gpt-3.5-turbo", history_context_len=1000)
    agent.generate = lambda x: ""  # so that the LLM is not triggered
    buffer = agent.in_context_learn(verbal_bandit, n_trajs=2)
    assert len(buffer) == 2
    assert len(buffer[0].verbal_prompts) == 10

    # Test save/load
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.json')
    temp_files.append(temp_file.name)
    temp_file.close()

    buffer.dump(temp_file.name)
    loaded_buffer = DatasetBuffer.load(temp_file.name)
    assert len(loaded_buffer) == 2


def test_cb_llm_rh_sampling(temp_files):
    # Test LLMCBAgentRH
    init_cb_env()
    agent = LLMCBAgentRH(verbal_env, "gpt-3.5-turbo", history_context_len=1000)
    agent.generate = lambda x: ""  # so that the LLM is not triggered
    buffer = agent.in_context_learn(verbal_env, n_trajs=2)
    assert len(buffer) == 2
    assert len(buffer[0].verbal_prompts) == 10

    # Test save/load
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.json')
    temp_files.append(temp_file.name)
    temp_file.close()

    buffer.dump(temp_file.name)
    loaded_buffer = DatasetBuffer.load(temp_file.name)
    assert len(loaded_buffer) == 2


def test_cb_llm_rh_with_ag_sampling(temp_files):
    # Test LLMCBAgentRHWithAG
    init_cb_env()
    ucb_guide = LinUCBGuide(LinUCBAgent(env))
    agent = LLMCBAgentRHWithAG(verbal_env, ucb_guide, "gpt-3.5-turbo", history_context_len=1000)
    agent.generate = lambda x: ""  # so that the LLM is not triggered
    buffer = agent.in_context_learn(verbal_env, n_trajs=2)
    assert len(buffer) == 2
    assert len(buffer[0].verbal_prompts) == 10
    assert buffer[0].action_info is not None

    # Test save/load
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.json')
    temp_files.append(temp_file.name)
    temp_file.close()

    buffer.dump(temp_file.name)
    loaded_buffer = DatasetBuffer.load(temp_file.name)
    assert len(loaded_buffer) == 2


def test_mab_llm_sh_with_ag_sampling(temp_files):
    # Setup environment
    core_bandit = BernoulliBandit(2, 10, [0.2, 0.5], 123)
    verbal_bandit = VerbalMultiArmedBandit(core_bandit, "VideoWatching")

    # Test LLMMABAgentSHWithAG
    ucb_guide = UCBGuide(UCBAgent(core_bandit))
    agent = LLMMABAgentSHWithAG(verbal_bandit, ucb_guide, "gpt-3.5-turbo", history_context_len=1000)
    agent.generate = lambda x: ""  # so that the LLM is not triggered
    buffer = agent.in_context_learn(verbal_bandit, n_trajs=2)
    assert len(buffer) == 2
    assert len(buffer[0].verbal_prompts) == 10
    assert buffer[0].action_info is not None

    # Test save/load
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.json')
    temp_files.append(temp_file.name)
    temp_file.close()

    buffer.dump(temp_file.name)
    loaded_buffer = DatasetBuffer.load(temp_file.name)
    assert len(loaded_buffer) == 2


def test_mab_llm_sh_with_ag_sampling(temp_files):
    # Setup environment
    core_bandit = BernoulliBandit(2, 10, [0.2, 0.5], 123)
    verbal_bandit = VerbalMultiArmedBandit(core_bandit, "VideoWatching")

    # Test LLMMABAgentSHWithAG
    ucb_guide = UCBGuide(UCBAgent(core_bandit))
    oracle = UCBAgent(core_bandit)
    agent = OracleLLMMABAgentSHWithAG(verbal_bandit, ucb_guide, oracle)

    agent.generate = lambda x: ""  # so that the LLM is not triggered
    buffer = agent.in_context_learn(verbal_bandit, n_trajs=2)
    assert len(buffer) == 2
    assert len(buffer[0].verbal_prompts) == 10
    assert buffer[0].action_info is not None

    # Test save/load
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.json')
    temp_files.append(temp_file.name)
    temp_file.close()

    buffer.to_sft_data(temp_file.name)
    data = buffer.to_sft_data()
    print(data[0][4])
