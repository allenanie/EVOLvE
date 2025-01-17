from banditbench.tasks.cb.movielens import MovieLens, MovieLensVerbal
from banditbench.tasks.mab.env import BernoulliBandit, GaussianBandit, VerbalMultiArmedBandit
from banditbench.tasks.mab import create_high_var_gaussian_bandit, create_low_var_gaussian_bandit, create_large_gap_bernoulli_bandit, create_small_gap_bernoulli_bandit

class SmallBench:
    def __init__(self):
        self.envs = [

        ]

    def create_envs(self, args):
        # we create 4 bernoulli tasks
        eval_envs = []
        eval_verbal_envs = []
        exp_id = 1
        for num_arms in [5, 20]:
            for domain in ['ClothesShopping']:
                for gap in ['easy', 'hard']:
                    if gap == 'easy':
                        bern_env = create_small_gap_bernoulli_bandit(num_arms, args.horizon, seed=exp_id)
                    else:
                        bern_env = create_large_gap_bernoulli_bandit(num_arms, args.horizon, seed=exp_id)

                    eval_envs.append(bern_env)

                    bern_verbal_env = VerbalMultiArmedBandit(bern_env, domain, scenario_seed=exp_id)

                    eval_verbal_envs.append(bern_verbal_env)

                    exp_id += 1

        return eval_envs, eval_verbal_envs