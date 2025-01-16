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

EVOLvE is a framework for evaluating Large Language Models (LLMs) for In-Context Reinforcement Learning (ICRL). We provide a flexible framework for single-step RL experiments (bandit) with LLMs. This repository contains the code to reproduce the results from the EVOLvE paper.

## 📰 News

- [Jan 2025] 🎉 EVOLvE codebase is released and available on [GitHub](https://github.com/allenanie/EVOLvE)
- [Jan 2025] 📦 First version of `banditbench` package is published on PyPI
- [Oct 2024] 📄 Our paper ["EVOLvE: Evaluating and Optimizing LLMs For Exploration"](https://arxiv.org/abs/2410.06238) is now available on arXiv

## 🚀 Features

- Flexible framework for evaluating LLMs for In-Context Reinforcement Learning (ICRL)
- Support for both multi-armed and contextual bandit scenarios
- Mixin-based design for highly customizable LLM agents
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

# this is a 5-armed bandit, it allows any agent to interact with it 100 times (horizon=100)
# in particular, this is a BernoulliBandit, which means the reward is sampled from a Bernoulli distribution
# For each arm, we set the probability of getting a reward to be [0.2, 0.2, 0.2, 0.2, 0.5]
core_bandit = BernoulliBandit(5, horizon=100, arm_params=[0.2, 0.2, 0.2, 0.2, 0.5])

# this is a verbal bandit, which wraps the core bandit and provides a verbal interface for the agent
# the scenario is "ClothesShopping", which means the agent will see actions as clothing item names like `shirt`, `pants`, `shoes`, etc.
verbal_bandit = VerbalMultiArmedBandit(core_bandit, "ClothesShopping")

# we create an agent that summarizes the history of interaction (the summary is using statistics -- not produced by an LLM)
# LLM uses the summary to make decisions
agent = LLMAgent.build(verbal_bandit, summary=True, model="gpt-3.5-turbo")

# we run the agent in-context learning on the verbal bandit for 5 trajectories
llm_result = agent.in_context_learn(verbal_bandit, n_trajs=5)

# we create a UCB agent, which is a classic agent that uses Upper Confidence Bound to make decisions
classic_agent = UCBAgent(core_bandit)

# we run the classic agent in-context learning on the core bandit for 5 trajectories
classic_result = classic_agent.in_context_learn(core_bandit, n_trajs=5)

classic_result.plot_performance(llm_result, labels=['UCB', 'GPT-3.5 Turbo'])
```

Doing this will give you a plot like this:

<p align="left">
  <img src="https://github.com/allenanie/EVOLvE/blob/main/assets/UCBvsLLM.png?raw=true" alt="UCB vs LLM" style="width: 40%;"/>
</p>


## 🌍 Environments & 🤖 Agents

(Add code example here)

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
(Add more specific instructions)

### Creating Custom Agents

(Add instructions for creating custom agents)

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
