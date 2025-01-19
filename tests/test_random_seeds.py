"""
We have a rather elaborate random seed strategy, controlled in the sampling process.
We test whether given the same master random seed (set in the env),
Both LLM and UCB can get the same reward if they choose the same action.
"""

from banditbench.tasks.mab import BernoulliBandit, VerbalMultiArmedBandit
from banditbench.agents.llm import LLMAgent
from banditbench.agents.classics import UCBAgent

def test_compare_env():
    core_bandit = BernoulliBandit(5, horizon=2, arm_params=[0.2, 0.2, 0.2, 0.2, 0.5], seed=321)
    verbal_bandit = VerbalMultiArmedBandit(core_bandit, "VideoWatching")

    agent = LLMAgent.build_with_env(verbal_bandit, summary=True, model="gpt-3.5-turbo")
    agent.generate = lambda x: "A"

    data = agent.in_context_learn(verbal_bandit, 5)

    classic_agent = UCBAgent(core_bandit)
    data_from_ucb = classic_agent.in_context_learn(core_bandit, 5)

    initial_reward = []
    for i in range(3):
        print(data.trajectories[i][0])
        initial_reward.append(data.trajectories[i][0].reward)

    initial_reward_ucb = []
    for i in range(3):
        print(data_from_ucb.trajectories[i][0])
        initial_reward_ucb.append(data_from_ucb.trajectories[i][0].reward)

    # write an assertion
    assert initial_reward == initial_reward_ucb