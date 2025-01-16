import tempfile
import os
import numpy as np
import pytest

from banditbench.sampling.sampler import DatasetBuffer, Trajectory
from banditbench.tasks.mab import BernoulliBandit, VerbalMultiArmedBandit

from banditbench.agents.classics import UCBAgent, LinUCBAgent
from banditbench.agents.llm import (LLMMABAgentRH, LLMMABAgentSH, LLMCBAgentRH, LLMCBAgentRHWithAG, LLMMABAgentSHWithAG,
                                    OracleLLMCBAgentRH, OracleLLMCBAgentRHWithAG, OracleLLMMABAgentSH,
                                    OracleLLMMAbAgentRH, OracleLLMMABAgentSHWithAG)

from banditbench.training.fewshot import FewShot
from banditbench.agents.guides import UCBGuide, LinUCBGuide

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


def mab_fake_steps(agent, verbal_bandit):
    verbal_bandit.reset()

    _, reward, done, info = verbal_bandit.step('A')

    action = info['interaction'].mapped_action
    agent.update(action, reward, info)

    _, reward, done, info = verbal_bandit.step('B')
    action = info['interaction'].mapped_action
    agent.update(action, reward, info)

    _, reward, done, info = verbal_bandit.step('A')

    action = info['interaction'].mapped_action
    agent.update(action, reward, info)

    _, reward, done, info = verbal_bandit.step('B')
    action = info['interaction'].mapped_action
    agent.update(action, reward, info)


def test_mab_fewshot(temp_files):
    core_bandit = BernoulliBandit(2, 10, [0.2, 0.5], 123)
    verbal_bandit = VerbalMultiArmedBandit(core_bandit, "VideoWatching")

    # Test LLMMABAgentRH
    agent = LLMMABAgentRH(verbal_bandit, "gpt-3.5-turbo", history_context_len=1000)
    agent.generate = lambda x: "Fake Action"

    buffer = agent.in_context_learn(verbal_bandit, n_trajs=2)
    assert len(buffer) == 2
    assert len(buffer[0].verbal_prompts) == 10

    # Test save/load
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.json')
    temp_files.append(temp_file.name)
    temp_file.close()

    buffer.dump(temp_file.name)

    # new agent
    agent = LLMMABAgentSH(verbal_bandit, "gpt-3.5-turbo", history_context_len=1000)

    fewshot_agent = FewShot.empower(agent, temp_file.name, 2)
    fewshot_agent.reset()

    mab_fake_steps(fewshot_agent, verbal_bandit)

    fewshot_agent.generate = lambda x: x

    constructed_llm_message = fewshot_agent.act()
    # assert "A video" in constructed_llm_message
    print(constructed_llm_message)


def test_mab_ag_fewshot(temp_files):
    core_bandit = BernoulliBandit(2, 10, [0.2, 0.5], 123)
    verbal_bandit = VerbalMultiArmedBandit(core_bandit, "VideoWatching")

    # Test LLMMABAgentRH
    ucb_guide = UCBGuide(UCBAgent(core_bandit))
    agent = LLMMABAgentSHWithAG(verbal_bandit, ucb_guide, "gpt-3.5-turbo", history_context_len=1000)
    agent.generate = lambda x: "Fake Action"

    buffer = agent.in_context_learn(verbal_bandit, n_trajs=2)
    assert len(buffer) == 2
    assert len(buffer[0].verbal_prompts) == 10

    # Test save/load
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.json')
    temp_files.append(temp_file.name)
    temp_file.close()

    buffer.dump(temp_file.name)
    # loaded_buffer = DatasetBuffer.load(temp_file.name)

    # new agent
    agent = LLMMABAgentSH(verbal_bandit, "gpt-3.5-turbo", history_context_len=1000)

    fewshot_agent = FewShot.empower(agent, temp_file.name, 2)
    fewshot_agent.reset()

    mab_fake_steps(fewshot_agent, verbal_bandit)

    fewshot_agent.generate = lambda x: x

    constructed_llm_message = fewshot_agent.act()
    # assert "A video" in constructed_llm_message
    print(constructed_llm_message)


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

def cb_fake_steps(agent,verbal_env):
    state, info = verbal_env.reset()

    new_state, reward, _, info = verbal_env.step(state, 'Star Wars (1977)')

    action = info['interaction'].mapped_action
    agent.update(state, action, reward, info)

    state = new_state

    new_state, reward, _, info = verbal_env.step(state, 'Fargo (1996)')
    action = info['interaction'].mapped_action
    agent.update(state, action, reward, info)

    state = new_state
    return state

def test_cb_fewshot(temp_files):

    # Test LLMMABAgentRH
    init_cb_env()
    agent = LLMCBAgentRH(verbal_env, "gpt-3.5-turbo", history_context_len=1000)
    agent.generate = lambda x: "Fake Action"

    buffer = agent.in_context_learn(verbal_env, n_trajs=2)
    assert len(buffer) == 2
    assert len(buffer[0].verbal_prompts) == 10

    # Test save/load
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.json')
    temp_files.append(temp_file.name)
    temp_file.close()

    buffer.dump(temp_file.name)

    # new agent
    agent = LLMCBAgentRH(verbal_env, "gpt-3.5-turbo", history_context_len=1000)

    fewshot_agent = FewShot.empower(agent, temp_file.name, 2)
    fewshot_agent.reset()
    state, _ = verbal_env.reset()

    cb_fake_steps(fewshot_agent, verbal_env)

    fewshot_agent.generate = lambda x: x

    constructed_llm_message = fewshot_agent.act(state)
    # assert "A video" in constructed_llm_message
    print(constructed_llm_message)

def test_cb_ag_fewshot(temp_files):

    # Test LLMMABAgentRH
    init_cb_env()
    guide = LinUCBGuide(LinUCBAgent(env))
    agent = LLMCBAgentRHWithAG(verbal_env, guide, "gpt-3.5-turbo", history_context_len=1000)
    agent.generate = lambda x: "Fake Action"

    buffer = agent.in_context_learn(verbal_env, n_trajs=2)
    assert len(buffer) == 2
    assert len(buffer[0].verbal_prompts) == 10

    # Test save/load
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.json')
    temp_files.append(temp_file.name)
    temp_file.close()

    buffer.dump(temp_file.name)

    # new agent
    agent = LLMCBAgentRH(verbal_env, "gpt-3.5-turbo", history_context_len=1000)

    fewshot_agent = FewShot.empower(agent, temp_file.name, 2)
    fewshot_agent.reset()
    state, _ = verbal_env.reset()

    cb_fake_steps(fewshot_agent, verbal_env)

    fewshot_agent.generate = lambda x: x

    constructed_llm_message = fewshot_agent.act(state)
    # assert "A video" in constructed_llm_message
    print(constructed_llm_message)
