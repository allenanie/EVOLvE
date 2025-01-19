from banditbench.agents.guides import ActionInfoField, UCBGuide, ThompsonSamplingGuide
from banditbench.tasks.mab.env import BernoulliBandit
from banditbench.agents.classics import UCBAgent, ThompsonSamplingAgent

def test_action_info_field():
    print()
    print(str(ActionInfoField("exploration bonus", 0.003) + ActionInfoField("exploitation value", 0.003)))
    print(str(ActionInfoField("exploration bonus", "inf") + ActionInfoField("exploitation value", "inf")))

def test_action_info():
    field1 = ActionInfoField("filed 1", 1, "action info=(semantic meaning for field 1={:.2f}")
    field2 = ActionInfoField("filed 2", 2, "semantic meaning for field 2={:.2f}")
    field3 = ActionInfoField("filed 3", 3, "semantic meaning for field 3={:.2f})")
    # this gives us: action info=(semantic meaning for field 1=1.00, semantic meaning for field 2=2.00, semantic meaning for field 3=3.00)
    print(str(field1 + field2 + field3))

def test_ucb_guide():
    core_bandit = BernoulliBandit(2, 10, [0.5, 0.2], 123)
    ucb_agent = UCBAgent(core_bandit)
    ucb_agent.reset()

    ucb_guide = UCBGuide(ucb_agent)
    print(ucb_guide.get_action_guide_info(0).to_str())

def test_ts_guide():
    core_bandit = BernoulliBandit(2, 10, [0.5, 0.2], 123)
    ts_agent = ThompsonSamplingAgent(core_bandit)
    ts_agent.reset()

    ts_guide = ThompsonSamplingGuide(ts_agent)
    ts_guide = ThompsonSamplingGuide(core_bandit)
    ts_guide = ThompsonSamplingGuide(core_bandit, alpha_prior=2.0)
    print(ts_guide.get_action_guide_info(0).to_str())

test_action_info_field()
test_action_info()
test_ucb_guide()