import tempfile
import os
import numpy as np
import pytest

from banditbench.sampling.sampler import DatasetBuffer, Trajectory
from banditbench.tasks.mab import BernoulliBandit, VerbalMultiArmedBandit

from banditbench.agents.classics import UCBAgent, LinUCBAgent
from banditbench.agents.llm import (LLMMABAgentRH, LLMMABAgentSH, LLMCBAgentRH, LLMCBAgentRHWithAG, LLMMABAgentSHWithAG,
                                    OracleLLMCBAgentRH, OracleLLMCBAgentRHWithAG, OracleLLMMAbAgentSH,
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

    buffer = agent.collect(verbal_bandit, n_trajectories=2)
    assert len(buffer) == 2
    assert len(buffer[0].verbal_prompts) == 10

    # Test save/load
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.json')
    temp_files.append(temp_file.name)
    temp_file.close()

    # buffer.dump(temp_file.name)
    # loaded_buffer = DatasetBuffer.load(temp_file.name)
    fewshot_agent = FewShot.empower(agent, buffer, 2)
    fewshot_agent.reset()

    mab_fake_steps(fewshot_agent, verbal_bandit)

    # print(buffer[0].verbal_prompts[2]['action_history'])
    # print(buffer[0].verbal_prompts[2]['decision_query'])
    # print(buffer[0].verbal_prompts[2]['label'])

    print(fewshot_agent.demos)
    fewshot_agent.generate = lambda x: x

    constructed_llm_message = fewshot_agent.act()
    # assert "A video" in constructed_llm_message
    print(constructed_llm_message)