from typing import List, Union

from banditbench.agents.classics import UCBAgent, LinUCBAgent
from banditbench.agents.llm import LLMAgent, LLMAgentBuilder
from banditbench.agents.guides import UCBGuide, LinUCBGuide
from banditbench.utils import calculate_cost

from banditbench.tasks import create_small_gap_bernoulli_bandit, VerbalMultiArmedBandit

class BaseBench:
    def __init__(self, num_threads=10):
        self.envs, self.verbal_envs = self.create_envs()
        self.n_trials = 20
        self.num_threads = num_threads

    def create_envs(self):
        raise NotImplementedError

    def evaluate(self, agents: List[LLMAgentBuilder], num_threads: int = None):
        """
        :param agents: LLMAgentBuilder is a list of partially initialized/configured agents, they are created through
                       LLMAgent.build() method (but did not pass in `env`)
        :return:
        """
        num_threads = self.num_threads if num_threads is None else num_threads
        env_to_agent_results = {}
        for i, env in enumerate(self.envs):
            assert hasattr(env, 'name'), "Environment must have a name attribute"
            env_to_agent_results[env.name] = {}

            for j, agent_builder in enumerate(agents):
                agent = agent_builder.build(env)
                assert hasattr(agent, 'name'), "Agent must have a name attribute"
                assert hasattr(agent, 'in_context_learn'), "Agent must have a in_context_learn method"
                print(f"Evaluating {agent.name} ({j}/{len(agents)}) on {env.name} ({i}/{len(self.envs)}) ")
                env_to_agent_results[env.name][agent.name] = agent.in_context_learn(env, n_trajs=self.n_trials,
                                                                                         num_threads=num_threads)

        return env_to_agent_results

    def calculate_eval_cost(self, model: Union[List[str], str] = 'gpt-3.5-turbo', verbose=False):
        # the way we calculate cost is by doing a data collection
        # we different Context Layer design: raw history, summarized history, summarized history + AG
        env_to_results = {}
        for i, (env, verbal_env) in enumerate(zip(self.envs, self.verbal_envs)):
            ucb_guide = UCBGuide(UCBAgent(env))
            oracle = UCBAgent(env)
            agent = LLMAgent.build_with_env(verbal_env, ucb_guide, oracle)
            assert "oracle" in agent.name.lower(), "The agent must be an oracle agent to calculate cost"
            print(f"Calculating {env.name} ({i}/{len(self.envs)}) ")

            env_to_results[verbal_env.name] = agent.in_context_learn(verbal_env, n_trajs=1, num_threads=1)

        models = [model] if isinstance(model, str) else model
        model_to_env_costs = {}
        
        for model_name in models:
            env_to_cost = {}
            for env_name, data_buffer in env_to_results.items():
                data = data_buffer.to_sft_data()
                cost = 0
                for step in data[0]:
                    cost += calculate_cost(model_name, step['prompt'], step['label'])
                env_to_cost[env_name] = cost * self.n_trials
            model_to_env_costs[model_name] = env_to_cost

        # Pretty print costs per model and environment
        if verbose:
            for model_name, env_to_cost in model_to_env_costs.items():
                print(f"\nCost Analysis for {model_name}")
                print("-" * 50)
                print(f"{'Environment':<30} {'Cost ($)':<10}")
                print("-" * 50)

                total_cost = 0
                for env_name, cost in env_to_cost.items():
                    print(f"{env_name:<30} ${cost:.4f}")
                    total_cost += cost

                print("-" * 50)
                print(f"{'Total Cost':<30} ${total_cost:.4f}")
                print(f"\nEstimated total cost for {len(self.envs)} environments "
                      f"with {self.n_trials} trials: ${total_cost:.2f}")
                
        # Print final summary table with just model names and total costs
        print("\nFinal Cost Summary")
        print("-" * 30)
        print(f"{'Model':<20} {'Total Cost ($)':<10}")
        print("-" * 30)
        # Sort models by total cost
        sorted_models = sorted(model_to_env_costs.items(),
                             key=lambda x: sum(x[1].values()))
        for model_name, env_to_cost in sorted_models:
            total = sum(env_to_cost.values())
            print(f"{model_name:<20} ${total:.2f}")
        print("-" * 30)

        return model_to_env_costs


class HardCoreBench(BaseBench):
    """
    If you include the highest performing model, you can use this set of tasks.
    It is sufficiently challenging, and does not cost too much to evaluate.
    1000 * 30 = 30000 calls for one model.

    We evaluate on a small set of tasks that are considered hard for the agent.
    This includes 4 MAB, Bernoulli, Hard case
    """

    def __init__(self, num_threads=10):
        super().__init__(num_threads=num_threads)
        self.envs, self.verbal_envs = self.create_envs()

    def create_envs(self):
        # we create 4 bernoulli tasks
        eval_envs = []
        eval_verbal_envs = []
        exp_id = 1
        for num_arms in [20]:
            for domain in ['VideoWatching', 'ClothesShopping']:
                for gap in ['hard']:
                    bern_env = create_small_gap_bernoulli_bandit(num_arms, horizon=1000, seed=exp_id)
                    eval_envs.append(bern_env)
                    bern_verbal_env = VerbalMultiArmedBandit(bern_env, domain, scenario_seed=exp_id)
                    eval_verbal_envs.append(bern_verbal_env)

                    exp_id += 1

        return eval_envs, eval_verbal_envs


class HardCorePlusBench:
    """
    Both Bernoulli and Gaussian, Hard cases
    """

    def __init__(self):
        self.envs, self.verbal_envs = self.create_envs()
        self.n_trials = 20

    def create_envs(self):
        # we create 4 bernoulli tasks
        eval_envs = []
        eval_verbal_envs = []
        exp_id = 1
        for num_arms in [20]:
            for domain in ['VideoWatching', 'ClothesShopping']:
                for gap in ['hard']:
                    bern_env = create_small_gap_bernoulli_bandit(num_arms, horizon=1000, seed=exp_id)
                    eval_envs.append(bern_env)
                    bern_verbal_env = VerbalMultiArmedBandit(bern_env, domain, scenario_seed=exp_id)
                    eval_verbal_envs.append(bern_verbal_env)

                    exp_id += 1

        return eval_envs, eval_verbal_envs


class MovieBench:
    """
    Evaluate on Contextual Bandit
    """
    pass


class FullBench:
    pass
