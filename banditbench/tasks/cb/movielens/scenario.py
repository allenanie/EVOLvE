import os
from typing import List
from banditbench.tasks.utils import dedent
from banditbench.tasks.scenario import BanditScenario, BanditConfig


# TODO: bad design, should allow people to load whatever few-shot traj they want, need to change
class CBConfig(BanditConfig):
    bandit_type: str
    domain: str
    num_arms: int

    def get_file_name(self):
        return f"bandit_{self.domain}_arms{self.num_arms}_trial0_fewshot_equal_space.json"

    def get_file_path(self, base_dir: str):
        return os.path.join(base_dir, self.get_file_name())


class MovieLensScenario(BanditScenario):
    name = 'movie'
    unit = 'movie'
    reward_unit = 'User returned a rating of'
    max_reward = '5'

    base_description = dedent(
        """You are an AI movie recommendation assistant for a streaming platform powered by a bandit algorithm that offers a wide variety of films from different studios and genres.
          There are {} unique movies you can recommend, named {}
          When a user visits the streaming platform, you assess their demographic description to choose a movie to suggest.
          You aim to match the user with movies they are most likely to watch and enjoy.
          Each time a user watches a recommended movie, you adjust your recommendation algorithms to better predict and meet future user preferences.
          Your goal is to enhance the user's viewing experience by providing personalized and engaging movie suggestions.
        """
    )

    detailed_description = dedent(
        """You are an AI movie recommendation assistant for a streaming platform powered by a bandit algorithm that offers a wide variety of films from different studios and genres.
          There are {} unique movies you can recommend, named {}
          When a user visits the streaming platform, you assess their demographic description to choose a movie to suggest.
          You aim to match the user with movies they are most likely to watch and enjoy.
          Each time a user watches a recommended movie, you adjust your recommendation algorithms to better predict and meet future user preferences.
          Your goal is to enhance the user's viewing experience by providing personalized and engaging movie suggestions.
          
          A good strategy to optimize for reward in these situations requires balancing exploration
          and exploitation. You need to explore to try out different movies and find those
          with high rewards, but you also have to exploit the information that you have to
          accumulate rewards.
        """
    )

    action_names: List[str]

    def get_instruction(self, version="base"):
        """We add the few-shot examples in here"""
        if version == 'base':
            prompt = self.base_description.format(
                len(self.action_names), '\n' + ',\n'.join(self.action_names) + '\n'
            )
        elif version == 'detailed':
            prompt = self.detailed_description.format(
                len(self.action_names), '\n' + ',\n'.join(self.action_names) + '\n'
            )
        else:
            raise ValueError(f'Unknown description version {version}')

        # this is where few-shot prompt is added
        if self.fewshot_examples != "":
            prompt += '\n' + self.fewshot_examples

        return prompt
