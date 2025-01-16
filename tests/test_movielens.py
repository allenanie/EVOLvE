from banditbench.tasks.cb.movielens import MovieLens, MovieLensVerbal

def test_construct_env():
    env = MovieLens('100k-ratings', 5, 20, 5, 'train', save_data_dir='./tensorflow_datasets/')
    state, _ = env.reset()
    verbal_env = MovieLensVerbal(env)
    verbal_state, info = verbal_env.reset()

test_construct_env()