import numpy as np
import scipy
import scipy.stats
import matplotlib.pyplot as plt

from typing import List, Dict


def dedent(text: str):
    """
    Remove leading and trailing whitespace for each line
    For example:
        ```
        Line 1 has no leading space
            Line 2 has two leading spaces
        ```
        The output will be :
        ```
        Line 1 has no leading space
        Line 2 has two leading spaces
        ```
    This allows writing cleaner multiline prompts in the code.
    """
    return "\n".join([line.strip() for line in text.split("\n")])


def plot_cumulative_reward(all_rewards: List[List[float]], horizon: int, title=None):
    """
    :param all_rewards: Should be [num_trials, horizon] -- we run the exploration N times, and each time the agent
                        has H interactions with the environment.
    :param horizon: The number of interactions the agent has with the environment.
    """

    reward_means, reward_sems = compute_cumulative_reward(all_rewards, horizon)
    plt.plot(reward_means)
    plt.fill_between(range(len(reward_means)), reward_means - reward_sems, reward_means + reward_sems, alpha=0.2)
    plt.ylabel("Average Reward Over Trials")
    plt.xlabel("Number of Interactions (Horizon)")
    if title is not None:
        plt.title(title)
    plt.show()


def compute_cumulative_reward(all_rewards: List[List[float]], horizon: int):
    all_rewards = np.vstack(all_rewards)
    cum_rewards = np.cumsum(all_rewards, axis=1)

    all_rewards = cum_rewards / np.arange(1, horizon + 1)

    reward_means = np.mean(all_rewards, axis=0)
    reward_sems = scipy.stats.sem(all_rewards, axis=0)

    return reward_means, reward_sems

def plot_multi_cumulative_reward(config_name_to_all_rewards: Dict[str, List[List[float]]], horizon: int, title=None):
    """
    :param config_name_to_all_rewards: Should be a dictionary of the form {config_name: all_rewards}
    :param horizon: The number of interactions the agent has with the environment.
    """

    for config_name, all_rewards in config_name_to_all_rewards.items():
        reward_means, reward_sems = compute_cumulative_reward(all_rewards, horizon)
        plt.plot(reward_means, label=config_name)
        plt.fill_between(range(len(reward_means)), reward_means - reward_sems, reward_means + reward_sems, alpha=0.2)
    plt.ylabel("Average Reward Over Trials")
    plt.xlabel("Number of Interactions (Horizon)")
    if title is not None:
        plt.title(title)
    plt.legend()
    plt.show()