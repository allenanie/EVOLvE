import numpy as np
from banditbench.tasks.mab.env import Interaction, VerbalInteraction
from banditbench.tasks.mab.env import BernoulliBandit, GaussianBandit, VerbalMultiArmedBandit


def create_small_gap_bernoulli_bandit(num_arms, horizon, seed=None):
    small_gap_delta = 0.2
    means = [0.5 - small_gap_delta / 2] * (num_arms - 1) + [0.5 + small_gap_delta / 2]
    core_bandit = BernoulliBandit(num_arms, horizon, means, seed)
    return core_bandit

def create_large_gap_bernoulli_bandit(num_arms, horizon, seed=None):
    big_gap_delta = 0.5
    means = [0.5 - big_gap_delta / 2] * (num_arms - 1) + [0.5 + big_gap_delta / 2]
    core_bandit = BernoulliBandit(num_arms, horizon, means, seed)
    return core_bandit

def create_high_var_gaussian_bandit(num_arms, horizon, seed=None):
    hard_variance = 3.
    rng = np.random.default_rng(seed)
    params = []
    for _ in range(num_arms):
        params.append((rng.normal(0, hard_variance), hard_variance))
    core_bandit = GaussianBandit(num_arms, horizon, params, seed)
    return core_bandit

def create_low_var_gaussian_bandit(num_arms, horizon, seed=None):
    easy_variance = 1.
    rng = np.random.default_rng(seed)
    params = []
    for _ in range(num_arms):
        params.append((rng.normal(0, easy_variance), easy_variance))
    core_bandit = GaussianBandit(num_arms, horizon, params, seed)
    return core_bandit