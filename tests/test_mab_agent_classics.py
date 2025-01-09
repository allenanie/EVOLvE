from banditbench.tasks.mab.env import BernoulliBandit
from banditbench.agents.classics import UCBAgent, ThompsonSamplingAgent


def test_ucb_agent():
    # construct it, and then print out the action guide
    core_bandit = BernoulliBandit(2, 10, [0.5, 0.2], 123)
    ucb_agent = UCBAgent(core_bandit)
    ucb_agent.reset()

    action = ucb_agent.select_arm()
    assert type(action) == int
    assert action < core_bandit.num_arms and action >= 0

    # print(ucb_agent.get_guide_info()[0].to_str())


def test_thompson_sampling_agent():
    core_bandit = BernoulliBandit(2, 10, [0.5, 0.2], 123)
    ts_agent = ThompsonSamplingAgent(core_bandit)
    ts_agent.reset()

    action = ts_agent.select_arm()
    assert type(action) == int
    assert action < core_bandit.num_arms and action >= 0

    # print(ts_agent.get_guide_info()[0].to_str())

test_ucb_agent()
test_thompson_sampling_agent()
