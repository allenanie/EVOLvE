# EVOLvE
The repo that stores the code to reproduce the result in EVOLvE paper

Requires `Python >= 3.9`

## Use Existing Multi-Armed Bandit Secnario

## Adding Custom Multi-Armed Bandit Secnario

Inherit from X class, and do Y...

## Use Contextual Bandit Scenario


## Representation of Decision-Making Context

We represent the decision making context into three segments:

```text
{Task Description + Instruction} (provided by the environment)
{History of interaction} (decided by the agent)
{Query prompt for the next decision} (provided by the environment)
```

For example, algorithm guide provides side information that is updated during each historical interaction, but it is also changed to make the next decision.


### Debug FAQ

1. There is some potential issue with `TFDS` (Tensorflow Datasets) library. If you open two Jupyter notebooks sharing the same kernel, and try to load the dataset -- even with different save location, the kernel might still die. Not clear what the immediate fix is.