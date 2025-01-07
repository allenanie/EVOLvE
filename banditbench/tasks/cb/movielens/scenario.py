from banditbench.tasks.utils import dedent
from banditbench.tasks.scenario import BanditScenario, BanditConfig

# TODO: check if this can inherit from BanditScenario
class MovieLensScenario:
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