import csv

from pydantic import BaseModel
from typing import Dict, Any, Tuple, Union, List, Optional

import os

from banditbench.tasks.env import State, Action, ExpectedReward, Bandit


class Interaction(BaseModel):
    state: State
    action: Action
    expected_reward: ExpectedReward
    is_random: Union[bool, None] = None

    def __init__(self, state: State, action: Action, expected_reward: ExpectedReward,
                 is_random: Union[bool, None] = None) -> None:
        super().__init__(state=state, action=action, expected_reward=expected_reward, is_random=is_random)


class ContextualBandit(Bandit):
    history: List[Interaction]

    def reward_fn(self, state: State, action: int) -> float:
        """In a contextual bandit, this is a function f(x, a)"""
        raise NotImplementedError

    def reset(self) -> State:
        raise NotImplementedError

    def step(self, state: State, action: Action) -> Tuple[State, float, bool, Dict[str, Any]]:
        raise NotImplementedError

# the step method can be written abstractly (because it just calls core_bandit)
class VerbalContextualBandit(ContextualBandit):
    pass


# Write it all here, but then break into
class MovieLensCB(ContextualBandit):
    def __init__(self,
                 task_name: str,
                 num_arms: int,
                 horizon: int,
                 rank_k: int = 10,
                 mode: str = 'eval',
                 seed: Optional[int] = None,
                 save_data_dir: Optional[str] = None) -> None:
        """
        task_name: str: the name of the dataset we load from the movielens directory
        rank_k: int: the dimension of the bandit arm context and user embedding. `rank_k` is only effective if it's smaller than num_arms.
                     rank_k = min(rank_k, num_arms)
        mode: str: 'eval' or 'train'. If 'train', we will load the training set, otherwise we load the test set.
        save_data_dir: point to the directory where the data is saved. If None, we default to Tensorflow Dataset.
        """

        assert mode in ['eval', 'train']
        assert '100k' in task_name or '1m' in task_name
        assert type(seed) == int or seed is None

        self.task_name = task_name
        self.num_arms = num_arms
        self.horizon = horizon
        self.rank_k = rank_k
        self.mode = mode
        self.seed = seed

        self.save_data_dir = save_data_dir

        self.history = []

        if self.mode == 'train' and '100k' in self.task_name:
            self.start_user_idx = 0
            self.end_user_idx = 100
        elif self.mode == 'eval' and '100k' in self.task_name:
            self.start_user_idx = 100
        elif self.mode == 'train' and '1m' in self.task_name:
            self.start_user_idx = 0
            self.end_user_idx = 200
        elif self.mode == 'eval' and '1m' in self.task_name:
            self.start_user_idx = 200

        self.initialize_defaults()

    def initialize_defaults(self) -> None:
        """Note this is not a reset function. It loads necessary files and processes them."""
        # https://github.com/scpike/us-state-county-zip/tree/master
        # load in csv "geo-data.csv" data from the data directory
        data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'movielens', 'data')
        self.zipdata = {}
        with open(os.path.join(data_dir, 'geo-data.csv')) as f:
            csvfile = csv.DictReader(f)
            for row in csvfile:
                self.zipdata[row['zipcode']] = ("{} of {}".format(row['city'], row['county']), row['state'])

        # now we run SVD for matrix completion
        # We shuffle the context/users

        ratings_df = load_data_files(self.task_name, self.save_data_dir)

        self._data_matrix, self._user_idx_to_feats, self._movie_idx_to_feats = (
            load_movielens_data(ratings_df, top_k_movies=self.num_arms, seed=self.seed)
        )

        self._num_users, self._num_movies = self._data_matrix.shape
        self.n_actions = self._num_movies
        self.d = self.rank_k

        # Compute the SVD.
        u, s, vh = np.linalg.svd(self._data_matrix, full_matrices=False)

        # Keep only the largest singular values.
        self._u_hat = u[:, :self.rank_k].astype(np.float32)
        self._s_hat = s[:self.rank_k].astype(np.float32)
        self._v_hat = np.transpose(vh[:self.rank_k]).astype(np.float32)

        self._approx_ratings_matrix = np.matmul(
            self._u_hat * self._s_hat, np.transpose(self._v_hat)
        )
        self.prev_user = None

    def _get_reward(self, action_idx):
        """Returns the reward (rating) for a given user-movie pair.

        Args:
            action_idx (int): a number between 0 - max_action, indicating which
              movie in the list.

        Returns:
            float: The approximated rating for the user-movie pair.
        """
        if self.prev_user is None:
            raise Exception('Need to use get_context() first')

        return self._approx_ratings_matrix[self.prev_user, action_idx]

    def get_context(self):
        """Returns the context for a given user.

        Returns:
            tuple: (svd_features, feat_dict)
        """
        if self.split == 'train':
            user_index = random.randint(self.start_user_idx, self.end_user_idx - 1)
        else:
            user_index = random.randint(self.start_user_idx, self._num_users - 1)

        # user_index = random.randint(0, self._num_users - 1)  # both ends included

        self.prev_user = user_index

        # Get user features from SVD
        svd_features = self._u_hat[user_index]
        user_features = self._user_idx_to_feats[user_index]

        feat_dict = {
            'oracle_feat': svd_features,
            'user_features': user_features,
            'user_index': user_index,
        }

        return svd_features, feat_dict

    def step(self, action):
        if not self.llm_readable:
            action = str(action)

        assert action in [
            str(i) for i in range(self.k_arms)
        ], f'Action {action} is not in the action space.'

        action_id = int(action)

        self.h += 1
        if self.h < self.horizon:
            done = False
        else:
            done = True

        reward = self._get_reward(action_id)
        self.actions_taken.append(action_id)
        self.avg_rewards.append(reward)

        # get a new observation, if not done
        obs, info = self.get_context()

        if self.llm_readable:
            obs = self.get_user_feat_text(obs, info['user_features'])

        # returns observation of a new user
        # reward for the previous user
        # whether we terminate, and extra info

        return obs, reward, done, info

    def reset(self, ctx_seed=None):
        self.avg_rewards = []
        self.actions_taken = []
        self.h = 0

        if ctx_seed is not None:
            # ctx_seed decides sampling order
            self.ctx_seed = ctx_seed
            random.seed(ctx_seed)

        obs, info = self.get_context()
        if self.llm_readable:
            obs = self.get_user_feat_text(obs, info['user_features'])

        return obs, info

    def get_actions_text(self, genre=True):
        """
        this function returns "options" for the scenario:
        "Movie_Name (year) (genre1|genre2|...)"
        """

        options = []
        for i in range(len(self._movie_idx_to_feats)):
            movie_name, genres = self._movie_idx_to_feats[i]
            # avoid "Empire Strikes Back, The"
            # should be "The Empire Strikes Back"

            if isinstance(movie_name, bytes):
                movie_name = movie_name.decode()

            # get rid of encoding issues
            if "b'" == movie_name[:2] and "'" == movie_name[-1]:
                movie_name = movie_name[2:-1]

            if ', The' in movie_name:
                movie_name = str('The ' + movie_name.replace(', The', ''))

            genre_text = ''
            genre_list = None
            if type(genres) == str:
                genres = parse_int_list(genres)
            else:
                genres = genres.tolist()
            for g in genres:
                genre_text += self.movie_genre_to_text[int(g)] + '|'

            if genre:
                options.append(f'{movie_name} ({genre_text[:-1]})')
            else:
                options.append(f'{movie_name}')

        return options

    def generate_occupation_sentence(self, occupation, age, age_bucket, gender):
        occupation = (
            occupation.decode('utf-8')
            if isinstance(occupation, bytes)
            else occupation
        )

        if "b'" == occupation[:2] and "'" == occupation[-1]:
            occupation = occupation[2:-1]

        if age == -1:
            age = age_bucket

        # Define occupation categories and their corresponding templates
        employment_occupations = [
            'doctor',
            'technician',
            'artist',
            'writer',
            'engineer',
            'administrator',
            'librarian',
            'scientist',
            'programmer',
            'educator',
            'executive',
            'salesman',
            'lawyer',
        ]

        special_cases = {
            'student': (
                'This person is a {age}-year-old {gender} studying as a student'
            ),
            'healthcare': (
                'This person is a {age}-year-old {gender} working in the healthcare'
                ' field'
            ),
            'entertainment': (
                'This person is a {age}-year-old {gender} working in the'
                ' entertainment field'
            ),
            'marketing': (
                'This person is a {age}-year-old {gender} working in the marketing'
                ' field'
            ),
            'retired': 'This person is a {age}-year-old {gender} who is retired.',
            'homemaker': (
                'This person is a {age}-year-old {gender} working as a homemaker'
            ),
            'other': (
                'This person is a {age}-year-old {gender} with an occupation'
                " classified as 'other'"
            ),
            'none': (
                'This person is a {age}-year-old {gender} with no current'
                ' occupation'
            ),
        }

        if occupation in employment_occupations:
            return (
                f'This person is a {age}-year-old {gender}, working as a {occupation}'
            )
        elif occupation in special_cases:
            return special_cases[occupation].format(age=age, gender=gender)
        else:
            return (
                f'This person is a {age}-year-old {gender} with an occupation of'
                f' {occupation}'
            )

    def get_user_feat_text(self, obs, user_features):
        user_description_zipcode_template = (
            """ and live in an area with the zip code {zip_code}."""
        )
        user_description_full = """ and live in {county} county, {state}."""

        feature_text = (
            ' The user has some numerical values that represent'
            ' their true implicit preference or taste for all movies: {}.'
        )

        def get_gender(is_male):
            return 'man' if is_male else 'woman'

        def format_description(data):
            age_bucket, raw_age, is_male, _, occupation, zip_code = data
            age_gender_occupation_text = self.generate_occupation_sentence(
                occupation, int(raw_age), int(age_bucket), get_gender(is_male)
            )

            if safe_decode(zip_code) in self.zipdata:
                county, state = self.zipdata[safe_decode(zip_code)]
                return (
                        age_gender_occupation_text
                        + user_description_full.format(
                    gender=get_gender(is_male),
                    age=int(raw_age),
                    age_bucket=int(age_bucket),
                    occupation=safe_decode(occupation),
                    county=county,
                    state=state,
                )
                        + feature_text.format(str([n for n in obs.tolist()]))
                )
            else:
                return (
                        age_gender_occupation_text
                        + user_description_zipcode_template.format(
                    gender=get_gender(is_male),
                    age=int(raw_age),
                    age_bucket=int(age_bucket),
                    occupation=safe_decode(occupation),
                    zip_code=safe_decode(zip_code),
                )
                        + feature_text.format(str([n for n in obs.tolist()]))
                )

        description = format_description(user_features)
        return description


class MovieLensVerbalCB(VerbalContextualBandit):
    pass


import numpy as np
import re
import tensorflow_datasets as tfds

MOVIELENS_NUM_USERS = 943
MOVIELENS_NUM_MOVIES = 1682

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
    # Try loading cached files first
    print("loading from tfds")
    if bandit_type == '100k-ratings':
        ratings = tfds.load('movielens/100k-ratings', data_dir=save_data_dir)
    elif bandit_type == '1m-ratings':
        ratings = tfds.load('movielens/1m-ratings', data_dir=save_data_dir)
    else:
        raise Exception('Not implemented yet')

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
