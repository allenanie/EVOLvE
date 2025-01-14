import re
import numpy as np

# Remove these two??
# MOVIELENS_NUM_USERS = 943
# MOVIELENS_NUM_MOVIES = 1682

movie_genre_to_text = [
    'Action',
    'Adventure',
    'Animation',
    'Children',
    'Comedy',
    'Crime',
    'Documentary',
    'Drama',
    'Fantasy',
    'Film-Noir',
    'Horror',
    'IMAX',
    'Musical',
    'Mystery',
    'Romance',
    'Sci-Fi',
    'Thriller',
    'Unknown',
    'War',
    'Western',
    '(no genres listed)',
]


def safe_decode(value, default='Unknown'):
    if isinstance(value, bytes):
        try:
            return value.decode('utf-8')
        except UnicodeDecodeError:
            try:
                return value.decode('latin-1')
            except UnicodeDecodeError:
                return default
    elif isinstance(value, str):
        try:
            return eval(value).decode('utf-8')
        except:
            return value
    return str(value)


def parse_int_list(s):
    # Remove square brackets and split by whitespace
    numbers = re.findall(r'-?\d+', s)

    # Convert strings to floats
    return [int(num) for num in numbers]


def load_data_files(bandit_type, save_data_dir=None):
    import tensorflow_datasets as tfds

    # Try loading cached files first
    print("loading from tfds")
    if bandit_type == '100k-ratings':
        ratings = tfds.load('movielens/100k-ratings', data_dir=save_data_dir)
    elif bandit_type == '1m-ratings':
        ratings = tfds.load('movielens/1m-ratings', data_dir=save_data_dir)
    else:
        ratings = tfds.load(bandit_type, data_dir=save_data_dir)
        raise ValueError(f"Other movielens datasets not verified yet: {bandit_type}")

    ratings_df = tfds.as_dataframe(ratings['train'])  # there is only one split
    print("data converted to pandas dataframe")

    return ratings_df


def load_movielens_data(ratings_df, top_k_movies=20, seed=1234):
    """Need to return three things:

    1.User index to preference mapping (need to start with an index)
    2. User index to user features
    3. Movie index to movie features

    The missing rating is filled in with 0.
    """
    unique_users = ratings_df['user_id'].unique()

    top_movies = ratings_df['movie_id'].value_counts()[:top_k_movies]
    # unique_movies = ratings_df['movie_id'].unique()
    unique_movies = top_movies.index.values

    # users are shuffled each time
    np.random.seed(seed)
    np.random.shuffle(unique_users)

    user_id_map = {id: i for i, id in enumerate(unique_users)}
    movie_id_map = {id: i for i, id in enumerate(unique_movies)}
    data_matrix = np.zeros((len(unique_users), len(unique_movies)))

    user_idx_to_feats = {}
    movie_idx_to_feats = {}

    for _, row in ratings_df.iterrows():
        i = user_id_map[row['user_id']]
        if row['movie_id'] in movie_id_map:
            j = movie_id_map[row['movie_id']]
            data_matrix[i, j] = row['user_rating']

            if j not in movie_idx_to_feats:
                movie_idx_to_feats[j] = [
                    row['movie_title'],
                    row['movie_genres'],
                ]

        if i not in user_idx_to_feats:
            if 'raw_user_age' in row:
                raw_user_age = row['raw_user_age']
            else:
                raw_user_age = -1

            user_idx_to_feats[i] = [
                row['bucketized_user_age'],
                raw_user_age,
                row['user_gender'],
                row['user_occupation_label'],
                row['user_occupation_text'],
                str(row['user_zip_code']),
            ]

    return data_matrix, user_idx_to_feats, movie_idx_to_feats
