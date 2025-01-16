import os
from typing import Optional
from banditbench.utils import dedent
from banditbench.tasks.scenario import MABScenario

class ScenarioUtil:
    """
    Inheriting from this class unifies the subclass' default __init__ method
    """
    def __init__(self, num_actions: int,
                 seed: Optional[int] = None):
        super().__init__(num_actions=num_actions, action_names=self.action_names, action_unit=self.action_unit, 
                         base_description=self.base_description, detailed_description=self.detailed_description,
                         query_prompt=self.query_prompt, seed=seed)

class ButtonPushing(ScenarioUtil, MABScenario):
  name = 'btn'
  
  # 100 actions
  action_names = [
      "blue",
      "green",
      "red",
      "yellow",
      "orange",
      "purple",
      "pink",
      "brown",
      "black",
      "white",
      "gray",
      "cyan",
      "magenta",
      "maroon",
      "olive",
      "lime",
      "teal",
      "navy",
      "indigo",
      "violet",
      "gold",
      "silver",
      "bronze",
      "coral",
      "turquoise",
      "lavender",
      "tan",
      "beige",
      "ivory",
      "aqua",
      "azure",
      "crimson",
      "fuchsia",
      "khaki",
      "peach",
      "plum",
      "sienna",
      "mint",
      "rose",
      "ruby",
      "salmon",
      "scarlet",
      "amber",
      "emerald",
      "jade",
      "mauve",
      "ochre",
      "periwinkle",
      "sapphire",
      "topaz",
      "chartreuse",
      "burgundy",
      "mustard",
      "cinnamon",
      "cobalt",
      "mint green",
      "forest green",
      "sky blue",
      "lilac",
      "tangerine",
      "raspberry",
      "pumpkin",
      "blush",
      "eggplant",
      "seafoam",
      "spruce",
      "lemon",
      "denim",
      "flamingo",
      "sand",
      "apricot",
      "honey",
      "chestnut",
      "midnight blue",
      "moss green",
      "bordeaux",
      "lavender blush",
      "slate gray",
      "gunmetal",
      "mint cream",
      "dark salmon",
      "persimmon",
      "cranberry",
      "wheat",
      "bistre",
      "magenta haze",
      "sepia",
      "ultramarine",
      "lime green",
      "steel blue",
      "blush pink",
      "pearl",
      "mulberry",
      "antique white",
      "burnt orange",
      "coral pink",
      "ice blue",
      "bright yellow",
      "honeydew",
      "baby blue",
  ]

  action_unit = "button"
  base_description = dedent(
      """You are a bandit algorithm in a room with {} buttons labeled {}.
                Each button is associated with a Bernoulli distribution with a fixed but unknown mean; the means for the two buttons could be different.
                For either button, when you press it, you will get a reward that is sampled from the button's associated distribution. Your goal is to maximize the total reward.
                """
  )

  detailed_description = dedent(
      """You are a bandit algorithm in a room with {} buttons labeled {}.
                Each button is associated with a Bernoulli distribution with a fixed but unknown mean; the means for the two buttons could be different.
                For either button, when you press it, you will get a reward that is sampled from the button's associated distribution. Your goal is to maximize the total reward .

                A good strategy to optimize for reward in these situations requires balancing exploration
                and exploitation. You need to explore to try out all of the buttons and find those
                with high rewards, but you also have to exploit the information that you have to
                accumulate rewards.
                """
  )

class OnlineAds(ScenarioUtil, MABScenario):
  name = 'ads'  

  action_names = [
      "A",
      "B",
      "C",
      "D",
      "E",
      "F",
      "G",
      "H",
      "I",
      "J",
      "K",
      "L",
      "M",
      "N",
      "O",
      "P",
      "Q",
      "R",
      "S",
      "T",
      "U",
      "V",
      "W",
      "X",
      "Y",
      "Z",
      "AA",
      "AB",
      "AC",
      "AD",
      "AE",
      "AF",
      "AG",
      "AH",
      "AI",
      "AJ",
      "AK",
      "AL",
      "AM",
      "AN",
      "AO",
      "AP",
      "AQ",
      "AR",
      "AS",
      "AT",
      "AU",
      "AV",
      "AW",
      "AX",
      "AY",
      "AZ",
      "BA",
      "BB",
      "BC",
      "BD",
      "BE",
      "BF",
      "BG",
      "BH",
      "BI",
      "BJ",
      "BK",
      "BL",
      "BM",
      "BN",
      "BO",
      "BP",
      "BQ",
      "BR",
      "BS",
      "BT",
      "BU",
      "BV",
      "BW",
      "BX",
      "BY",
      "BZ",
      "CA",
      "CB",
      "CC",
      "CD",
      "CE",
      "CF",
      "CG",
      "CH",
      "CI",
      "CJ",
      "CK",
      "CL",
      "CM",
      "CN",
      "CO",
      "CP",
      "CQ",
      "CR",
      "CS",
      "CT",
      "CU",
      "CV",
      "CW",
      "CX",
      "CY",
      "CZ",
  ]
  action_unit = "advertisement"
  base_description = dedent(
      """You are recommendation engine powered by a bandit algorithm that chooses advertisements to display to
                users when they visit your webpage. There are {} advertisements you can choose from,
                named {}. When a user visits the webpage you can choose an advertisement
                to display and you will observe whether the user clicks on the ad or not. You model
                this by assuming that each advertisement has a certain click rate and users click on
                advertisements with their corresponding rates."""
  )

  detailed_description = dedent(
      """You are recommendation engine powered by a bandit algorithm that chooses advertisements to display to
                users when they visit your webpage. There are {} advertisements you can choose from,
                named {}. When a user visits the webpage you can choose an advertisement
                to display and you will observe whether the user clicks on the ad or not. You model
                this by assuming that each advertisement has a certain click rate and users click on
                advertisements with their corresponding rates.

                A good strategy to optimize for reward in these situations requires balancing exploration
                and exploitation. You need to explore to try out all of the advertisements and find those
                with high rewards, but you also have to exploit the information that you have to
                accumulate rewards.
                """
  )


class VideoWatching(ScenarioUtil, MABScenario):
  name = 'vid'
  action_names = [
      "A",
      "B",
      "C",
      "D",
      "E",
      "F",
      "G",
      "H",
      "I",
      "J",
      "K",
      "L",
      "M",
      "N",
      "O",
      "P",
      "Q",
      "R",
      "S",
      "T",
      "U",
      "V",
      "W",
      "X",
      "Y",
      "Z",
      "AA",
      "AB",
      "AC",
      "AD",
      "AE",
      "AF",
      "AG",
      "AH",
      "AI",
      "AJ",
      "AK",
      "AL",
      "AM",
      "AN",
      "AO",
      "AP",
      "AQ",
      "AR",
      "AS",
      "AT",
      "AU",
      "AV",
      "AW",
      "AX",
      "AY",
      "AZ",
      "BA",
      "BB",
      "BC",
      "BD",
      "BE",
      "BF",
      "BG",
      "BH",
      "BI",
      "BJ",
      "BK",
      "BL",
      "BM",
      "BN",
      "BO",
      "BP",
      "BQ",
      "BR",
      "BS",
      "BT",
      "BU",
      "BV",
      "BW",
      "BX",
      "BY",
      "BZ",
      "CA",
      "CB",
      "CC",
      "CD",
      "CE",
      "CF",
      "CG",
      "CH",
      "CI",
      "CJ",
      "CK",
      "CL",
      "CM",
      "CN",
      "CO",
      "CP",
      "CQ",
      "CR",
      "CS",
      "CT",
      "CU",
      "CV",
      "CW",
      "CX",
      "CY",
      "CZ",
  ]
  action_unit = "video"
  base_description = dedent(
      """You are a video recommendation system powered by a bandit algorithm for an online streaming platform. 
    There are {} videos available in your library, titled {}. 
    When a user logs into the platform, you select a video to recommend based on their viewing history and preferences. 
    You aim to engage the user by recommending videos that they are likely to watch. 
    Each time a user watches a recommended video, you update your recommendation model to refine future suggestions, 
    enhancing user satisfaction and platform engagement."""
  )

  detailed_description = dedent(
      """You are a video recommendation system powered by a bandit algorithm for an online streaming platform. 
    There are {} videos available in your library, titled {}. 
    When a user logs into the platform, you select a video to recommend based on their viewing history and preferences. 
    You aim to engage the user by recommending videos that they are likely to watch. 
    Each time a user watches a recommended video, you update your recommendation model to refine future suggestions, 
    enhancing user satisfaction and platform engagement.

    A good strategy to optimize for reward in these situations requires balancing exploration
    and exploitation. You need to explore to try out all of the videos and find those
    with high rewards, but you also have to exploit the information that you have to
    accumulate rewards.
    """
  )

class ClothesShopping(ScenarioUtil, MABScenario):
  name = 'clothes'
  
  action_names = [
      "Velvet Vogue Jacket",
      "Silk Serenity Dress",
      "Urban Mystique Jeans",
      "Celestial Symphony Scarf",
      "Retro Revival Sneakers",
      "Ethereal Elegance Blouse",
      "Midnight Mirage Trousers",
      "Vintage Vibe Coat",
      "Opulent Oasis Gown",
      "Mystic Mosaic Shirt",
      "Chic Cascade Skirt",
      "Radiant Reverie Blazer",
      "Bohemian Bliss Boots",
      "Stellar Style Suit",
      "Luminous Legend Jacket",
      "Panache Paradise Pants",
      "Gilded Galaxy Dress",
      "Posh Phoenix Pullover",
      "Enchanté Enigma Ensemble",
      "Luxe Loft Leggings",
      "Bespoke Bliss Blouse",
      "Dapper Dreams Denim",
      "Glamour Gaze Gown",
      "Eminent Essence Earrings",
      "Avant-Garde Allure Anklets",
      "Refined Radiance Romper",
      "Sophisticated Silhouette Sweater",
      "Opulent Oasis Overcoat",
      "Noble Nimbus Nightwear",
      "Divine Debonair Dungarees",
      "Regal Reverie Raincoat",
      "Marvelous Mirage Mittens",
      "Extravagant Expanse Earmuffs",
      "Ingenious Impressions Iridescent",
      "Prestige Paradigm Parka",
      "Dazzling Dynasty Dress",
      "Supreme Sylvan Sandals",
      "Elite Echoes Espadrilles",
      "Sublime Sashay Scarf",
      "Opulent Odyssey Overalls",
      "Elegant Eclipse Ensemble",
      "Refined Renaissance Robe",
      "Lavish Luminary Legwarmers",
      "Sculpted Serenity Shawl",
      "Timeless Titan Turtleneck",
      "Whimsical Whimsy Wrap",
      "Piquant Prestige Poncho",
      "Noble Nexus Nightshirt",
      "Faithful Fantasy Frock",
      "Radiant Realm Rainboots",
      "Alluring Apex Apron",
      "Delightful Dusk Dress",
      "Prestigious Prism Pants",
      "Gossamer Gleam Gloves",
      "Supreme Spectrum Slippers",
      "Celestial Chic Cape",
      "Magnetic Melody Mittens",
      "Glacial Grace Gaiters",
      "Gleaming Galore Gown",
      "Infinite Impeccable Jacket",
      "Viridian Visage Vest",
      "Ornamental Oasis Overalls",
      "Enchanted Eden Earmuffs",
      "Aesthetic Aura Anorak",
      "Mystical Muse Midi",
      "Exquisite Expanse Earrings",
      "Relished Realm Romper",
      "Glamorous Gala Gown",
      "Serenade Spirit Shawl",
      "Charmed Cascade Cardigan",
      "Splendid Solstice Sweater",
      "Majestic Manor Moccasins",
      "Titanic Tempest Tunic",
      "Royal Reverie Wrap",
      "Dazzling Delight Dress",
      "Verdant Vogue Vest",
      "Ethereal Expanse Earrings",
      "Sublime Symphony Sweater",
      "Pristine Plaid Pullover",
      "Opulent Oasis Outerwear",
      "Fabled Flora Frock",
      "Glittering Gala Gloves",
      "Eminent Envy Earwarmer",
      "Classic Couture Cardigan",
      "Mythic Mirage Muffler",
      "Silk Spectrum Slip",
      "Refined Rascal Rainjacket",
      "Splendid Sorcery Socks",
      "Lustrous Lattice Leggings",
      "Radiant Ripple Romper",
      "Stellar Sheen Shawl",
      "Serenade Spirit Suit",
      "Opal Obsession Overcoat",
      "Bejeweled Bloom Blazer",
      "Ornate Oratorio Onesie",
      "Dapper Dusk Duffle Coat",
      "Prismatic Paradise Poncho",
      "Majestic Moonlit Mantle",
      "Vivid Velocity Vestment",
      "Radiant Radiance Rucksack",
  ]

  action_unit = "item"
  base_description = dedent(
      """You are an AI fashion assistant for an online boutique powered by a bandit algorithm that offers a variety of clothing options from different brands. 
    There are {} unique clothing items you can recommend, named {}. 
    When a customer visits the online store, you assess their style preferences and shopping history to choose an item to suggest. 
    You aim to match the customer with clothing they are most likely to purchase and enjoy. 
    Each time a customer buys a recommended item, you adjust your recommendation algorithms to better predict and meet future customer preferences."""
  )

  detailed_description = dedent(
      """You are an AI fashion assistant for an online boutique powered by a bandit algorithm that offers a variety of clothing options from different brands. 
    There are {} unique clothing items you can recommend, named {}. 
    When a customer visits the online store, you assess their style preferences and shopping history to choose an item to suggest. 
    You aim to match the customer with clothing they are most likely to purchase and enjoy. 
    Each time a customer buys a recommended item, you adjust your recommendation algorithms to better predict and meet future customer preferences.

    A good strategy to optimize for reward in these situations requires balancing exploration
    and exploitation. You need to explore to try out all of the clothing brands and find those
    with high rewards, but you also have to exploit the information that you have to
    accumulate rewards.
    """
  )
