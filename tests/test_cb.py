import numpy as np
from banditbench.tasks.types import State
from banditbench.tasks.cb.movielens import MovieLens, MovieLensVerbal

def test_save_state():
    print()
    s = State(feature=[1, 2, 3], index=1, info=None)
    print(s.model_dump_json())
    s = State(feature=np.array([1, 2, 3]), index=1, info={'field1': 1})
    print(s.model_dump_json())
    s = State(feature=np.int32(3), index=1, info={'field1': np.array([1,2,3])})
    print(s.model_dump_json())

test_save_state()

env = None
verbal_env = None


def init_cb_env():
    global env
    global verbal_env

    if env is None:
        env = MovieLens('100k-ratings', num_arms=5, horizon=200, rank_k=5, mode='train',
                        save_data_dir='./tensorflow_datasets/')
    if verbal_env is None:
        verbal_env = MovieLensVerbal(env)


def test_action_name():
    init_cb_env()

    state, info = verbal_env.reset()
    new_state, reward, _, info = verbal_env.step(state, 'Star Wars (1977) (Action|Adventure|Romance|Sci-Fi|War)')
    assert info['interaction'].is_random is False
