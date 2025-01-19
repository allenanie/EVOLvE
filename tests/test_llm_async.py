from banditbench.tasks.mab import BernoulliBandit, VerbalMultiArmedBandit
from banditbench.agents.llm import LLMAgent


def test_async():
    try:
        core_bandit = BernoulliBandit(5, horizon=20, arm_params=[0.2, 0.2, 0.2, 0.2, 0.5])
        verbal_bandit = VerbalMultiArmedBandit(core_bandit, "ClothesShopping")
        agent = LLMAgent.build_with_env(verbal_bandit, summary=True, model="gpt-3.5-turbo")
        data = agent.in_context_learn(verbal_bandit, 5)
    except Exception as e:
        print(e)
        print("Omit the LLM Async test")


if __name__ == '__main__':
    test_async()
