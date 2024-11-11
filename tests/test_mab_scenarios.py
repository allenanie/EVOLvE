from banditbench.tasks.mab_scenarios import ButtonPushing


def test_button_pushing():
    scenario = ButtonPushing(5)
    print(scenario.get_instruction("base"))
    print()
    print(scenario.get_instruction("detailed"))

test_button_pushing()