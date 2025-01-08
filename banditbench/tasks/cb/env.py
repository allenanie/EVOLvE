import csv

from pydantic import BaseModel
from pydantic.dataclasses import dataclass
from typing import Dict, Any, Tuple, Union, List, Optional

import os
import numpy as np

from banditbench.tasks.env import Action, ExpectedReward, Bandit

Info = Union[Dict[str, Any], None]


class State(BaseModel):
    feature: Any  # numpy array
    index: Union[int, None]  # a pointer to the dataset (if there is a dataset)
    info: Info = None  # additional information


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

    def sample_state(self) -> State:
        """
        We sample a state from the state distribution
        """
        raise NotImplementedError

    def reward_fn(self, state: State, action: Action) -> float:
        """In a contextual bandit, this is a function f(x, a)"""
        raise NotImplementedError

    def reset(self) -> Tuple[State, Info]:
        raise NotImplementedError

    def step(self, state: State, action: Action) -> Tuple[State, float, bool, Dict[str, Any]]:
        raise NotImplementedError

    @property
    def verbal_info(self) -> Dict[str, Any]:
        """
        CB might be able to provide additional information from the dataset about the state
        This property is used by the VerbalContextualBandit
        :return:
        """
        raise NotImplementedError


# the step method can be written abstractly (because it just calls core_bandit)
class VerbalContextualBandit(ContextualBandit):
    def __init__(self, core_bandit, *args, **kwargs):
        self.core_bandit = core_bandit

    @property
    def name(self) -> str:
        # cb_1m-ratings_arms10
        return self.core_bandit.name


from banditbench.tasks.cb.movielens.processing import load_data_files, load_movielens_data, movie_genre_to_text, \
    parse_int_list, safe_decode


# Write it all here, but then break into
class MovieLens(ContextualBandit):
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
        self.set_seed(seed)

        self.save_data_dir = save_data_dir

        self.history = []

        if '100k' in self.task_name:
            self.start_user_idx = 0
            self.end_user_idx = 100
        elif '1m' in self.task_name:
            self.start_user_idx = 0
            self.end_user_idx = 200

        self.initialize_defaults()

        self._verbal_info = {
            'user_idx_to_feats': self._user_idx_to_feats,
            'movie_idx_to_feats': self._movie_idx_to_feats,
            'movie_genre_to_text': movie_genre_to_text
        }

    def initialize_defaults(self) -> None:
        """Note this is not a reset function. It loads necessary files and processes them."""

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

    @property
    def name(self) -> str:
        # cb_1m-ratings_arms10
        return f"cb_{self.task_name}_arms{self.num_arms}_mode_{self.mode}"

    @property
    def verbal_info(self) -> Dict[str, Any]:
        """
        CB might be able to provide additional information from the dataset about the state
        :return:
        """
        return self._verbal_info

    def reward_fn(self, state: State, action: Action) -> float:
        """In a contextual bandit, this is a function f(x, a)"""
        user_index = state.index
        return self._approx_ratings_matrix[user_index, action]

    def sample_state(self) -> State:
        """Returns the context for a given user.

        Returns:
            tuple: (svd_features, feat_dict)
        """
        if self.mode == 'train':
            user_index = self.np_random.integers(self.start_user_idx, self.end_user_idx).item()
        else:
            user_index = self.np_random.integers(self.end_user_idx, self._num_users).item()

        # Get user features from SVD
        svd_features = self._u_hat[user_index]
        user_features = self._user_idx_to_feats[user_index]

        # this user_feature is a partial information feature
        # the svd user feature is the full information feature
        feat_dict = {
            'user_features': user_features
        }

        state = State(feature=svd_features, index=user_index, info=feat_dict)

        return state

    def step(self, state: State, action: Action):
        """Core step function that takes integer action and returns vector observation"""
        self.h += 1
        if self.h < self.horizon:
            done = False
        else:
            done = True

        reward = self.reward_fn(state, action)
        self.actions_taken.append(action)
        self.avg_rewards.append(reward)

        obs = self.sample_state()

        return obs, reward, done, {}

    def reset(self, ctx_seed=None) -> Tuple[State, Info]:
        self.avg_rewards = []
        self.actions_taken = []
        self.h = 0

        if ctx_seed is not None:
            # ctx_seed decides sampling order
            self.ctx_seed = ctx_seed
            self.set_seed(ctx_seed)

        obs = self.sample_state()

        return obs, None

from banditbench.tasks.cb.movielens.scenario import MovieLensScenario, CBConfig

class MovieLensVerbal(VerbalContextualBandit):
    def __init__(self,
                 core_bandit: MovieLens,
                 # ===== arguments for bandit_scenario_cls =====
                 instruction_type: str = "detailed",
                 num_fewshot: int = 0, few_shot_config: Optional[CBConfig] = None
                 ) -> None:
        self.core_bandit = core_bandit
        self.instruction_type = instruction_type
        self.bandit_scenario = MovieLensScenario(action_names=self.get_actions_text(),
                                                 num_actions=self.core_bandit.num_arms,
                                                 num_fewshot=num_fewshot,
                                                 few_shot_config=few_shot_config,
                                                 seed=None)  # no action shuffling
        self.initialize_defaults()

    def initialize_defaults(self) -> None:
        # https://github.com/scpike/us-state-county-zip/tree/master
        # load in csv "geo-data.csv" data from the data directory
        data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'movielens', 'data')
        self.zipdata = {}
        with open(os.path.join(data_dir, 'geo-data.csv')) as f:
            csvfile = csv.DictReader(f)
            for row in csvfile:
                self.zipdata[row['zipcode']] = ("{} of {}".format(row['city'], row['county']), row['state'])

    def step(self, state: State, action: Action) -> Tuple[State, float, bool, Dict[str, Any]]:
        """Step function that handles string/int actions and text observations"""

        # can add action validation here (instead of taking in a number)
        assert action in [
            str(i) for i in range(self.core_bandit.num_arms)
        ], f'Action {action} is not in the action space.'

        action_id = int(action)

        obs, reward, done, info = self.core_bandit.step(action_id)
        text = self.get_user_feat_text(obs.feature, obs.info['user_features'])
        obs.feature = text

        return obs, reward, done, info

    def reset(self, ctx_seed=None) -> Tuple[State, Info]:

        obs, _ = self.core_bandit.reset()
        text = self.get_user_feat_text(obs.feature, obs.info['user_features'])
        obs.feature = text

        verbal_instruction = self.bandit_scenario.get_instruction(self.instruction_type)

        return obs, {'instruction': verbal_instruction}

    def get_actions_text(self, genre=True):
        """
        this function returns "options" for the scenario:
        "Movie_Name (year) (genre1|genre2|...)"
        """

        options = []
        movie_idx_to_feats = self.core_bandit.verbal_info['movie_idx_to_feats']
        movie_genre_to_text = self.core_bandit.verbal_info['movie_genre_to_text']

        for i in range(len(movie_idx_to_feats)):
            movie_name, genres = movie_idx_to_feats[i]
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
                genre_text += movie_genre_to_text[int(g)] + '|'

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
