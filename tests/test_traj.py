from banditbench.tasks.cb.movielens import MovieLens
from banditbench.tasks.mab import BernoulliBandit

def test_mab_traj():
    core_bandit = BernoulliBandit(4, 500, [0.25, 0.25, 0.25, 0.75])
    core_bandit.step(2)
    core_bandit.step(3)

    traj = core_bandit.history[0] + core_bandit.history[1]
    print(traj)
    print(traj.model_dump())
    print(traj.model_dump_json())

def test_cb_traj():
    env = MovieLens('100k-ratings', num_arms=5, horizon=200, rank_k=5, mode='train', save_data_dir='./tensorflow_datasets/')
    obs, _ = env.reset()
    env.step(obs, 3)
    env.step(obs, 4)
    traj = env.history[0] + env.history[1]
    print(traj)
    print(traj.model_dump())
    print(traj.model_dump_json())

test_mab_traj()
test_cb_traj()