import numpy as np
import scipy
import scipy.stats
import matplotlib.pyplot as plt

from typing import List, Dict
from litellm.cost_calculator import completion_cost

plt.rcParams.update(
    {
        "font.size": 13,  # Default font size for text
        "axes.titlesize": 16,  # Font size for axes titles
        "axes.labelsize": 14,  # Font size for axes labels
        "xtick.labelsize": 13,  # Font size for x-tick labels
        "ytick.labelsize": 13,  # Font size for y-tick labels
        "legend.fontsize": 13,  # Font size for legend
        "figure.titlesize": 16,  # Font size for figure title
        "axes.formatter.useoffset": False,
        "axes.formatter.offset_threshold": 1,
    }
)
plt.rcParams["font.family"] = "DeJavu Serif"
plt.rcParams["font.serif"] = ["Times New Roman"]


def compute_pad(total_elements, K, desired_pad=5):
    # equal spaced sampling for few-shot in-context learning
    available_space = total_elements - K
    pad = min(desired_pad, available_space)
    return max(0, pad)


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


def compute_cumulative_reward(all_rewards: List[List[float]], horizon: int):
    all_rewards = np.vstack(all_rewards)
    cum_rewards = np.cumsum(all_rewards, axis=1)

    all_rewards = cum_rewards / np.arange(1, horizon + 1)

    reward_means = np.mean(all_rewards, axis=0)
    reward_sems = scipy.stats.sem(all_rewards, axis=0)

    return reward_means, reward_sems


def plot_cumulative_reward(all_rewards: List[List[float]], horizon: int, title=None, filename=None):
    """
    Plot cumulative reward over time with confidence interval.
    
    Args:
        all_rewards: List of reward sequences [num_trials, horizon] 
        horizon: Number of timesteps
        title: Optional plot title
    """
    reward_means, reward_sems = compute_cumulative_reward(all_rewards, horizon)

    plt.figure(figsize=(8, 6))

    # Plot main line
    plt.plot(range(horizon), reward_means, color='#2ecc71', linewidth=2.5)

    # Add confidence interval
    plt.fill_between(
        range(horizon),
        reward_means - reward_sems,
        reward_means + reward_sems,
        alpha=0.2,
        color='#2ecc71'
    )

    # Add grid
    plt.grid(True, linestyle='--', alpha=0.7)

    # Customize axes
    plt.ylabel("Average Reward Over Trials")
    plt.xlabel("Number of Interactions (Horizon)")

    if title is not None:
        plt.title(title, pad=20)

    # Customize appearance    
    plt.tight_layout()
    if filename is not None:
        plt.savefig(filename, dpi=120, bbox_inches='tight')

    plt.show()


def plot_multi_cumulative_reward(config_name_to_all_rewards: Dict[str, List[List[float]]], horizon: int, title=None, filename=None):
    """
    Plot multiple cumulative reward curves with confidence intervals.
    
    Args:
        config_name_to_all_rewards: Dict mapping config names to reward sequences
        horizon: Number of timesteps
        title: Optional plot title
    """

    plt.figure(figsize=(8, 6))

    # Plot lines for each config
    for i, (config_name, all_rewards) in enumerate(config_name_to_all_rewards.items()):
        reward_means, reward_sems = compute_cumulative_reward(all_rewards, horizon)

        # Plot main line
        plt.plot(range(horizon), reward_means, label=config_name, linewidth=2.5)

        # Add confidence interval
        plt.fill_between(
            range(horizon),
            reward_means - reward_sems,
            reward_means + reward_sems,
            alpha=0.2,
        )

    # Add grid
    plt.grid(True, linestyle='--', alpha=0.7)

    # Customize axes
    plt.ylabel("Average Reward Over Trials")
    plt.xlabel("Number of Interactions (Horizon)")

    if title is not None:
        plt.title(title, fontweight='bold', pad=20)

    # Customize appearance
    plt.legend(loc='lower right')
    plt.tight_layout()
    if filename is not None:
        plt.savefig(filename, dpi=120, bbox_inches='tight')

    plt.show()

def calculate_cost(model, input_seq, output_seq):
    """
    Calculate the cost in dollars for a given sequence of tokens.

    Returns:
        dict: The input and output costs (in dollars).
    """
    return completion_cost(model=model, prompt=input_seq, completion=output_seq)