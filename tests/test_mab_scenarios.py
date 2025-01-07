from banditbench.tasks.mab.scenarios import ButtonPushing

def test_button_pushing():
    scenario = ButtonPushing(5)
    print()
    print(scenario.get_instruction("base"))
    print()
    print(scenario.get_instruction("detailed"))

test_button_pushing()