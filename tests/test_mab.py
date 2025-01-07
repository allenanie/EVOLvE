from banditbench.tasks.mab.env import VerbalMultiArmedBandit, BernoulliBandit


def test_verbal_bandit(verbose=False):
    core_bandit = BernoulliBandit(2, 10, [0.5, 0.2], 123)
    verbal_bandit = VerbalMultiArmedBandit(core_bandit, "VideoWatching")
    assert len(verbal_bandit.history) == 0
    assert len(verbal_bandit.core_bandit.history) == 0

    query, reward, done, info = verbal_bandit.step("button 1")
    if verbose:
        print(query)
        print(reward)
        print(done)
        print(info)

def test_verbal_bandit_random():
    core_bandit = BernoulliBandit(2, 10, [0.5, 0.2], 123)
    verbal_bandit = VerbalMultiArmedBandit(core_bandit, "VideoWatching")
    assert len(verbal_bandit.history) == 0
    assert len(verbal_bandit.core_bandit.history) == 0

    _, _, _, info = verbal_bandit.step("button 1")
    assert info["is_random"] == True

def test_verbal_bandit_non_random():
    core_bandit = BernoulliBandit(2, 10, [0.5, 0.2], 123)
    verbal_bandit = VerbalMultiArmedBandit(core_bandit, "VideoWatching")
    action = verbal_bandit.action_names[0]
    _, _, _, info = verbal_bandit.step(action)
    assert info["is_random"] == False

def test_verbal_bandit_seed():
    core_bandit = BernoulliBandit(4, 10, [0.5, 0.2, 0.2, 0.2], 222)
    print(core_bandit.arm_params)
    verbal_bandit = VerbalMultiArmedBandit(core_bandit, "VideoWatching", scenario_seed=333)

    assert len(verbal_bandit.history) == 0
    assert len(verbal_bandit.core_bandit.history) == 0

    action = verbal_bandit.action_names[0]
    print(verbal_bandit.action_names)
    _, _, _, info = verbal_bandit.step(action)
    assert info["is_random"] == False

def test_verbal_bandit_history(verbose=False):
    core_bandit = BernoulliBandit(2, 10, [0.5, 0.2], 123)
    verbal_bandit = VerbalMultiArmedBandit(core_bandit, "VideoWatching")
    action = verbal_bandit.action_names[0]
    for _ in range(10):
        verbal_bandit.step(action)
    if verbose:
        print(verbal_bandit.history)
        print(len(verbal_bandit.history))
        print(len(verbal_bandit.core_bandit.history))
    assert len(verbal_bandit.history) == 10

def test_fewshot_loading():
    raise NotImplementedError

test_verbal_bandit()
test_verbal_bandit_random()
test_verbal_bandit_non_random()
test_verbal_bandit_seed()
test_verbal_bandit_history(verbose=True)