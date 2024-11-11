from banditbench.mab import VerbalBandit, BernoulliBandit
from banditbench.algs.classics import UCBAgent, GreedyAgent, ThompsonSamplingAgent, ActionInfo, ActionInfoField


def test_action_info_field():
    print(str(ActionInfoField("exploration bonus", "0.003") + ActionInfoField("exploitation value", "0.003")))

def test_action_info():
    field1 = ActionInfoField("filed 1", '1', "action info=(semantic meaning for field 1={:.2f}")
    field2 = ActionInfoField("filed 2", '2', "semantic meaning for field 2={:.2f}")
    field3 = ActionInfoField("filed 3", '3', "semantic meaning for field 3={:.2f})")
    # this gives us: action info=(semantic meaning for field 1=1.00, semantic meaning for field 2=2.00, semantic meaning for field 3=3.00) 
    print(str(field1 + field2 + field3))


def test_ucb_agent():
    # construct it, and then print out the action guide
    core_bandit = BernoulliBandit(2, 10, [0.5, 0.2], 123)
    ucb_agent = UCBAgent(core_bandit)
    ucb_agent.reset()

    action = ucb_agent.select_arm()
    assert type(action) == int
    assert action < core_bandit.num_arms and action >= 0

    print(ucb_agent.get_guide_info()[0].to_str())


def test_thompson_sampling_agent():
    core_bandit = BernoulliBandit(2, 10, [0.5, 0.2], 123)
    ts_agent = ThompsonSamplingAgent(core_bandit)
    ts_agent.reset()

    action = ts_agent.select_arm()
    assert type(action) == int
    assert action < core_bandit.num_arms and action >= 0

    print(ts_agent.get_guide_info()[0].to_str())



test_action_info_field()
test_ucb_agent()
test_thompson_sampling_agent()
