<div align="center">

# EVOLvE: Evaluating and Optimizing LLMs For Exploration In-Context

<p align="center">
  <img src="https://github.com/allenanie/EVOLvE/blob/main/assets/logo.png?raw=true" alt="EVOLvE Logo" width="200" height="200"/>
</p>


[![Github](https://img.shields.io/badge/EVOLvE-000000?style=for-the-badge&logo=github&logoColor=000&logoColor=white)](https://github.com/allenanie/EVOLvE)  [![ArXiv](https://img.shields.io/badge/EVOLvE-CF4545?style=for-the-badge&logo=arxiv&logoColor=000&logoColor=white)](https://arxiv.org/pdf/2410.06238)


[![PyPI version](https://badge.fury.io/py/banditbench.svg)](https://badge.fury.io/py/banditbench)
[![Python](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Build Status](https://github.com/allenanie/evolve/actions/workflows/python-app.yml/badge.svg)](https://github.com/allenanie/evolve/actions)

<div align="center" style="font-family: Arial, sans-serif;">
  <p>
    <a href="#-news" style="text-decoration: none; font-weight: bold;">🎉 News</a> •
    <a href="#️-installation" style="text-decoration: none; font-weight: bold;">✨ Getting Started</a> •
    <a href="#-features" style="text-decoration: none; font-weight: bold;">📖 Introduction</a>
  </p>
  <p>
    <a href="#-bandit-scenario-example" style="text-decoration: none; font-weight: bold;">🔧 Usage</a> •
    <a href="#-citation" style="text-decoration: none; font-weight: bold;">🎈 Citation</a> •
    <a href="#-acknowledgement" style="text-decoration: none; font-weight: bold;">🌻 Acknowledgement</a>
  </p>
</div>

</div>

EVOLvE is a framework for evaluating Large Language Models (LLMs) for In-Context Reinforcement Learning (ICRL). 
We provide a flexible framework for experimenting with different LLM Agent Context Layers and analyzing how they affect a model's ability to interact with RL environments (bandits). This repository contains the code to reproduce results from our EVOLvE paper.

## 📰 News

- [Jan 2025] 🎉 EVOLvE codebase is released and available on [GitHub](https://github.com/allenanie/EVOLvE)
- [Jan 2025] 📦 First version of `banditbench` package is published on PyPI
- [Oct 2024] 📄 Our paper ["EVOLvE: Evaluating and Optimizing LLMs For Exploration"](https://arxiv.org/abs/2410.06238) is now available on arXiv

## 🚀 Features

- Flexible framework for evaluating LLMs for In-Context Reinforcement Learning (ICRL)
- Support for both multi-armed and contextual bandit scenarios
- Mixin-based design for LLM agents with customizable **Context Layers**
- Built-in support for few-shot learning and demonstration
- Includes popular benchmark environments (e.g., MovieLens)


## 🛠️ Installation

### Option 1: Install from PyPI (Recommended for Users)

```bash
pip install banditbench
```

### Option 2: Install from Source (Recommended for Developers)

```bash
git clone https://github.com/allenanie/EVOLvE.git
cd EVOLvE
pip install -e .  # Install in editable mode for development
```

## 🎯 Bandit Scenario

We provide two types of bandit scenarios:

**Multi-Armed Bandit Scenario**
  - Classic exploration-exploitation problem with stochastic reward sampled from a fixed distributions
  - Agent learns to select the best arm without any contextual information
  - Example: Choosing between 5 different TikTok videos to show, without knowing which one is more popular at first

**Contextual Bandit Scenario**
  - Reward distributions depend on a context (e.g., user features)
  - Agent learns to map contexts to optimal actions
  - Example: Recommending movies to users based on their age, location (e.g., suggesting "The Dark Knight" to a 25-year-old who enjoys action movies and lives in an urban area)

<p align="center">
  <img src="https://github.com/allenanie/EVOLvE/blob/main/assets/bandit_scenario.png?raw=true" alt="Bandit Scenario Example"/>
</p>

## 🎮 Quick Start

### Evaluate LLMs for their In-Context Reinforcement Learning Performance

In this example, we will compare the performance of two agents (LLM and one of the classic agents) on a multi-armed bandit task.

```python
from banditbench.tasks.mab import BernoulliBandit, VerbalMultiArmedBandit
from banditbench.agents.llm import LLMAgent
from banditbench.agents.classics import UCBAgent

# this is a 5-armed bandit
# with the probability of getting a reward to be [0.2, 0.2, 0.2, 0.2, 0.5]
core_bandit = BernoulliBandit(5, horizon=100, arm_params=[0.2, 0.2, 0.2, 0.2, 0.5])

# The scenario is "ClothesShopping", agent sees actions as clothing items
verbal_bandit = VerbalMultiArmedBandit(core_bandit, "ClothesShopping")

# we create an LLM agent that uses summary statistics (mean, number of times, etc.)
agent = LLMAgent.build_with_env(verbal_bandit, summary=True, model="gpt-3.5-turbo")

llm_result = agent.in_context_learn(verbal_bandit, n_trajs=5)

# we create a UCB agent, which is a classic agent that uses 
# Upper Confidence Bound to make decisions
classic_agent = UCBAgent(core_bandit)

# we run the classic agent in-context learning on the core bandit for 5 trajectories
classic_result = classic_agent.in_context_learn(core_bandit, n_trajs=5)

classic_result.plot_performance(llm_result, labels=['UCB', 'GPT-3.5 Turbo'])
```

Doing this will give you a plot like this:

<p align="left">
  <img src="https://github.com/allenanie/EVOLvE/blob/main/assets/UCBvsLLM.png?raw=true" alt="UCB vs LLM" style="width: 60%;"/>
</p>

## 💰 Evaluation Cost

Each of the benchmark has a cost estimation tool for the inference cost. The listed cost is in $ amount which contains
all trials and repetitions.

```python
from banditbench import HardCoreBench, HardCorePlusBench, FullBench, CoreBench, MovieBench
bench = HardCoreBench()
cost = bench.calculate_eval_cost([
    'gemini-1.5-pro',
    'gemini-1.5-flash',
    'gpt-4o-2024-11-20',
    "gpt-4o-mini-2024-07-18",
    "o1-2024-12-17",
    "o1-mini-2024-09-12",
    "claude-3-5-sonnet-20241022",
    "claude-3-5-haiku-20241022"
])
```

You can evaluate an agent by doing:
```python
from banditbench.agents.llm import LLMAgent
from banditbench.agents.guide import UCBGuide

env_to_agent_results = bench.evaluate([
  LLMAgent.build(),  # Raw History Context Layer
  LLMAgent.build(summary=True),  # Summary Context Layer
  LLMAgent.build(summary=True, guide=UCBGuide(env))  # Summary + UCB Guide Context Layer
])
```

Cost estimation is performed for a **single** agent with raw history (the longest context). If you evaluate multiple agent,
you can simply multiply this cost by the number of agents.

| Model                     | Core     | HardCore     | HardCore+     | Full      | MovieBench     |
|---------------------------|----------|--------------|---------------|-----------|----------------|
| **gemini-1.5-flash**      | **$31.05**   | **$14.91**       | **$39.18**        | **$83.44**    | **$31.05**         |
| gpt-4o-mini-2024-07-18    | $62.10   | $29.83       | $78.36        | $166.88   | $62.10         |
| claude-3-5-haiku-20241022 | $414.33  | $198.97      | $522.64       | $1113.18  | $414.33        |
| **gemini-1.5-pro**        | **$517.54**  | **$248.55**      | **$652.98**       | **$1390.69**  | **$517.54**        |
| gpt-4o-2024-11-20         | $1035.07 | $497.11      | $1305.96      | $2781.38  | $1035.07       |
| o1-mini-2024-09-12        | $1242.09 | $596.53      | $1567.16      | $3337.66  | $1242.09       |
| claude-3-5-sonnet-20241022| $1243.00 | $596.91      | $1567.91      | $3339.53  | $1243.00       |
| o1-2024-12-17             | $6210.45 | $2982.64     | $7835.79      | $16688.31 | $6210.45       |

## 🌍 Environments & 🤖 Agents

Here are a list of agents that are supported by EVOLvE:

For Multi-Armed Bandit Scenario:

| Agent Name | Code | Interaction History | Algorithm Guide |
|------------|------|---------------------|-----------------|
| UCB | `UCBAgent(env)` | `False` | `NA` |
| Greedy | `GreedyAgent(env)` | `False` | `NA` |
| Thompson Sampling | `ThompsonSamplingAgent(env)` | `False` | `NA` |
| LLM with Raw History | `LLMAgent.build(env)` | `False` | `False` |
| LLM with Summary | `LLMAgent.build(env, summary=True)` | `True` | `False` |
| LLM with UCB Guide | `LLMAgent.build(env, summary=True, guide=UCBGuide(env))` | `True` | `True` |

For Contextual Bandit Scenario:

| Agent Name | Code | Interaction History | Algorithm Guide |
|------------|------|---------------------|-----------------|
| LinUCB | `LinUCBAgent(env)` | `False` | `NA` |
| LLM with Raw History | `LLMAgent.build(env)` | `False` | `False` |
| LLM with UCB Guide | `LLMAgent.build(env, guide=LinUCBGuide(env))` | `True` | `True` |

Here are a list of environments that are supported by EVOLvE:

**Multi-Armed Bandit Scenario**

| Environment Name | Code | Description |
|------------|------|-----------------|
| Bernoulli Bandit | `BernoulliBandit(n_arms, horizon, arm_params)` | Arm parameter is Bernoulli p|
| Gaussian Bandit | `GaussianBandit(n_arms, horizon, arm_params)` | Arm parameter is a tuple of (mean, variance)|

For LLM, we provide a `VerbalMultiArmedBandit` environment that converts the core bandit into a verbal bandit.

| Scenario Name | Code | Action Names |
|------------|------|-----------------|
| Button Pushing | `ButtonPushing` | Action names are colored buttons like "Red", "Blue", "Green", etc.|
| Online Ads | `OnlineAds` | Action names are online ads like "Ad A", "Ad B", "Ad C", etc.|
| Video Watching | `VideoWatching` | Action names are videos like "Video A", "Video B", "Video C", etc.|
| Clothes Shopping | `ClothesShopping` | Action names are clothing items like "Velvet Vogue Jacket", "Silk Serenity Dress", etc.|

They can be coupled together like:

```python
from banditbench.tasks.mab import BernoulliBandit, VerbalMultiArmedBandit

core_bandit = BernoulliBandit(2, 10, [0.5, 0.2], 123)
verbal_bandit = VerbalMultiArmedBandit(core_bandit, "VideoWatching")
```

**Contextual Bandit Scenario**

| Environment Name | Code | Description |
|------------|------|-----------------|
| MovieLens | `MovieLens(task_name, num_arms, horizon)` | `task_name` loads in specific MovieLens dataset|
| MovieLensVerbal | `MovieLensVerbal(env)` | Similar to VerbalEnv before. Scenario is fixed to be "MovieLens"|

```python
from banditbench.tasks.contextual import MovieLens, MovieLensVerbal

env = MovieLens('100k-ratings', num_arms=10, horizon=200, rank_k=5, mode='train',
                        save_data_dir='./tensorflow_datasets/')
verbal_env = MovieLensVerbal(env)
```

To use the environments listed in the paper, you can use the following code:

```python
from banditbench.tasks.mab import create_small_gap_bernoulli_bandit, create_large_gap_bernoulli_bandit
from banditbench.tasks.mab import create_high_var_gaussian_bandit, create_low_var_gaussian_bandit

easy_bern_bandit = create_small_gap_bernoulli_bandit(num_arms=5, horizon=1000)
```

## 🧩 Architecture

### Decision-Making Context

The framework represents decision-making contexts in three segments:

```text
{Task Description + Instruction} (provided by the environment)
{Few-shot demonstrations from historical interactions}
{Current history of interaction} (decided by the agent)
{Query prompt for the next decision} (provided by the environment)
```

### LLM Agents

We use a Mixin-based design pattern to provide maximum flexibility and customization options for agent implementation. This allows you to:
- Combine different agent behaviors
- Customize prompt engineering strategies
- Implement new decision-making algorithms


## 🔧 Customization

### Adding Custom Multi-Armed Bandit Scenarios

To create a custom bandit scenario:
1. Inherit from the base scenario class
2. Implement required methods
(Coming soon)

### Creating Custom Agents

(Coming soon)

## ⚠️ Known Issues

1. **TFDS Issues**: There is a known issue with TensorFlow Datasets when using multiple Jupyter notebooks sharing the same kernel. The kernel may crash when loading datasets, even with different save locations.

2. **TensorFlow Dependency**: The project currently requires TensorFlow due to TFDS usage. We plan to remove this dependency in future releases.

## 🎈 Citation

If you find EVOLvE useful in your research, please consider citing our paper:

```bibtex
@article{nie2024evolve,
  title={EVOLvE: Evaluating and Optimizing LLMs For Exploration},
  author={Nie, Allen and Su, Yi and Chang, Bo and Lee, Jonathan N and Chi, Ed H and Le, Quoc V and Chen, Minmin},
  journal={arXiv preprint arXiv:2410.06238},
  year={2024}
}
```

## 📄 License

This project is licensed under the [LICENSE NAME] - see the [LICENSE](LICENSE) file for details.

## 🌻 Acknowledgement

The design of EVOLvE is inspired by the following projects:

- [DSPy](https://github.com/stanfordnlp/dspy) 
- [Trace](https://github.com/microsoft/Trace)
- [Textgrad](https://github.com/zou-group/textgrad)
- [d3rlpy](https://d3rlpy.readthedocs.io/en/v2.6.0/)
- [Scala Mixin Trait](https://docs.scala-lang.org/tour/mixin-class-composition.html)
- [In-Context Reinforcement Learning Paper List](https://github.com/dunnolab/awesome-in-context-rl)

## 🤝 Contributing

We welcome contributions! Please start by reporting an issue or a feature request.

<p align="center">
  <img src="https://github.com/allenanie/EVOLvE/blob/main/assets/main.jpeg?raw=true" alt="EVOLvE Framework Overview"/>
</p>
