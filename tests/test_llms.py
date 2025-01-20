import re

from banditbench.tasks.mab import BernoulliBandit, VerbalMultiArmedBandit
from banditbench.tasks.cb.movielens import MovieLens, MovieLensVerbal

from banditbench.agents.guides import VerbalGuideBase, UCBGuide, LinUCBGuide
from banditbench.agents.classics import UCBAgent, LinUCBAgent

from banditbench.agents.llm import (LLMMABAgentRH, LLMMABAgentSH, LLMCBAgentRH, LLMCBAgentRHWithAG, LLMMABAgentSHWithAG,
                                    OracleLLMCBAgentRH, OracleLLMCBAgentRHWithAG, OracleLLMMABAgentSH,
                                    OracleLLMMAbAgentRH, OracleLLMMABAgentSHWithAG, LLMAgent)

core_bandit = BernoulliBandit(2, 10, [0.5, 0.2], 123)
verbal_bandit = VerbalMultiArmedBandit(core_bandit, "VideoWatching")

env = None
verbal_env = None


def init_cb_env():
    global env
    global verbal_env

    if env is None:
        env = MovieLens('100k-ratings', num_arms=5, horizon=200, rank_k=5, mode='train',
                        save_data_dir='./tensorflow_datasets/')
    if verbal_env is None:
        verbal_env = MovieLensVerbal(env)


def mab_fake_steps(agent):
    _, reward, done, info = verbal_bandit.step('A')

    action = info['interaction'].mapped_action
    agent.update(action, reward, info)

    _, reward, done, info = verbal_bandit.step('B')
    action = info['interaction'].mapped_action
    agent.update(action, reward, info)


def test_mab_agent_rh():
    agent = LLMMABAgentRH(verbal_bandit)
    agent.generate = lambda x: x  # so that the LLM is not triggered

    mab_fake_steps(agent)

    constructed_llm_message = agent.act()
    assert "A video" in constructed_llm_message
    print(constructed_llm_message)


def test_mab_agent_sh():
    agent = LLMMABAgentSH(verbal_bandit)
    agent.generate = lambda x: x  # so that the LLM is not triggered

    mab_fake_steps(agent)
    constructed_llm_message = agent.act()
    assert "A video" in constructed_llm_message
    assert "B video" in constructed_llm_message

    assert "1 times" in constructed_llm_message
    assert "average reward" in constructed_llm_message

    print(constructed_llm_message)


def extract_values(text):
    pattern = r'average reward ([\d.]+).*?exploitation value ([\d.]+)'
    matches = re.finditer(pattern, text)

    results = []
    for match in matches:
        avg_reward = float(match.group(1))
        exploit_value = float(match.group(2))
        results.append({
            'average_reward': avg_reward,
            'exploitation_value': exploit_value
        })
    return results


def equal_pulls_equal_bonus(text):
    # Extract all video data with regex
    pattern = r'(\w+) video, (\d+) times.*?exploration bonus ([\d.]+)'
    matches = re.finditer(pattern, text)

    # Store data in format: {num_pulls: [(video_name, exploration_bonus)]}
    pulls_data = {}

    # Parse and group by number of pulls
    for match in matches:
        video_name = match.group(1)
        num_pulls = int(match.group(2))
        exploration_bonus = float(match.group(3))

        if num_pulls not in pulls_data:
            pulls_data[num_pulls] = []
        pulls_data[num_pulls].append((video_name, exploration_bonus))

    # Check each group of videos with same pulls
    inconsistencies = []
    for pulls, videos in pulls_data.items():
        if len(videos) > 1:  # If there are multiple videos with same pulls
            first_bonus = videos[0][1]
            for video_name, bonus in videos[1:]:
                if bonus != first_bonus:
                    inconsistencies.append(
                        f"Inconsistency found: Videos with {pulls} pulls have different bonuses. "
                        f"{videos[0][0]} video: {videos[0][1]}, {video_name} video: {bonus}"
                    )

    # Return test results
    if not inconsistencies:
        return True, "All videos with equal pulls have equal exploration bonuses."
    else:
        return False, "\n".join(inconsistencies)


def test_mab_agent_sh_with_ag():
    guide = UCBGuide(UCBAgent(core_bandit))
    agent = LLMMABAgentSHWithAG(verbal_bandit, guide)
    agent.generate = lambda x: x  # so that the LLM is not triggered

    mab_fake_steps(agent)
    constructed_llm_message = agent.act()
    results = extract_values(constructed_llm_message)

    for result in results:
        assert result['average_reward'] == result['exploitation_value']

    pass_flag, message = equal_pulls_equal_bonus(constructed_llm_message)
    assert pass_flag, message
    print(constructed_llm_message)


def cb_fake_steps(agent):
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


def test_cb_agent_rh():
    init_cb_env()
    agent = LLMCBAgentRH(verbal_env)
    agent.generate = lambda x: x  # so that the LLM is not triggered

    state = cb_fake_steps(agent)

    constructed_llm_message = agent.act(state)
    print(constructed_llm_message)


def test_cb_agent_sh_with_ag():
    init_cb_env()
    guide = LinUCBGuide(LinUCBAgent(env))
    agent = LLMCBAgentRHWithAG(verbal_env, guide)
    agent.generate = lambda x: x  # so that the LLM is not triggered

    state = cb_fake_steps(agent)

    constructed_llm_message = agent.act(state)
    print(constructed_llm_message)

# need to test reset!!!!!
def test_mab_agent_rh_reset():
    agent = LLMMABAgentRH(verbal_bandit)
    agent.generate = lambda x: x  # so that the LLM is not triggered

    mab_fake_steps(agent)

    agent.reset()
    assert len(agent.interaction_history) == 0

def test_mab_agent_sh_reset():
    agent = LLMMABAgentSH(verbal_bandit)
    agent.generate = lambda x: x  # so that the LLM is not triggered

    mab_fake_steps(agent)
    agent.reset()

def test_mab_agent_sh_with_ag_reset():
    guide = UCBGuide(UCBAgent(core_bandit))
    agent = LLMMABAgentSHWithAG(verbal_bandit, guide)
    agent.generate = lambda x: x  # so that the LLM is not triggered

    mab_fake_steps(agent)

    agent.reset()

    # fully reset both agent and the AlgorithmGuide's agent
    assert len(agent.interaction_history) == 0
    assert agent.ag.agent.arms[0] == 0

def test_cb_agent_rh_reset():
    init_cb_env()
    agent = LLMCBAgentRH(verbal_env)
    agent.generate = lambda x: x  # so that the LLM is not triggered

    state = cb_fake_steps(agent)

    agent.reset()

    assert len(agent.interaction_history) == 0

def test_cb_agent_sh_with_ag_reset():
    init_cb_env()
    guide = LinUCBGuide(LinUCBAgent(env))
    agent = LLMCBAgentRHWithAG(verbal_env, guide)
    agent.generate = lambda x: x  # so that the LLM is not triggered

    state = cb_fake_steps(agent)

    agent.reset()

    assert len(agent.interaction_history) == 0
    assert len(agent.ag_info_history) == 0
    assert agent.ag.agent.b[0][0] == 0

def test_general_llm_agent_construction():
    # Test MAB agents
    agent = LLMAgent.build_with_env(verbal_bandit)
    assert isinstance(agent, LLMMABAgentRH)

    agent = LLMAgent.build_with_env(verbal_bandit, summary=True)
    assert isinstance(agent, LLMMABAgentSH)

    guide = UCBGuide(UCBAgent(core_bandit))
    agent = LLMAgent.build_with_env(verbal_bandit, guide)
    assert isinstance(agent, LLMMABAgentSHWithAG)

    oracle = UCBAgent(core_bandit)
    agent = LLMAgent.build_with_env(verbal_bandit, oracle)
    assert isinstance(agent, OracleLLMMAbAgentRH)

    agent = LLMAgent.build_with_env(verbal_bandit, guide, oracle)
    assert isinstance(agent, OracleLLMMABAgentSHWithAG)

    # Test CB agents
    init_cb_env()
    agent = LLMAgent.build_with_env(verbal_env)
    assert isinstance(agent, LLMCBAgentRH)

    guide = LinUCBGuide(LinUCBAgent(env))
    agent = LLMAgent.build_with_env(verbal_env, guide)
    assert isinstance(agent, LLMCBAgentRHWithAG)

    oracle = LinUCBAgent(env)
    agent = LLMAgent.build_with_env(verbal_env, oracle)
    assert isinstance(agent, OracleLLMCBAgentRH)

    agent = LLMAgent.build_with_env(verbal_env, guide, oracle)
    assert isinstance(agent, OracleLLMCBAgentRHWithAG)

def test_llm_agent_builder_construction():
    # Test MAB agents
    agent = LLMAgent.build()
    agent = agent.build_with_env(verbal_bandit)
    assert isinstance(agent, LLMMABAgentRH)

    agent = LLMAgent.build(summary=True)
    agent = agent.build_with_env(verbal_bandit)
    assert isinstance(agent, LLMMABAgentSH)

    guide = UCBGuide(core_bandit)
    agent = LLMAgent.build(guide, summary=True)
    agent = agent.build_with_env(verbal_bandit)
    assert isinstance(agent, LLMMABAgentSHWithAG)

    oracle = UCBAgent(core_bandit)
    agent = LLMAgent.build(oracle)
    agent = agent.build_with_env(verbal_bandit)
    assert isinstance(agent, OracleLLMMAbAgentRH)

    agent = LLMAgent.build(guide, oracle)
    agent = agent.build_with_env(verbal_bandit)
    assert isinstance(agent, OracleLLMMABAgentSHWithAG)