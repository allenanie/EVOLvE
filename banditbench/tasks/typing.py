from typing import Dict, Any, Tuple, Union, List, Optional
from pydantic import BaseModel, field_serializer
import numpy as np
import json

Action = Union[int, str]
ExpectedReward = Union[float, None]
Info = Union[Dict[str, Any], None]


def safe_json_encode(obj):
    try:
        return json.dumps(obj)
    except:
        return None


class State(BaseModel):
    feature: Any  # must be numpy array
    feature_text: Union[str, None] = None  # a text representation of the feature
    index: Union[int, None]  # a pointer to the dataset (if there is a dataset)
    info: Info = None  # additional information

    @field_serializer('info', mode='plain')
    def serialize_info(self, info: Info, _info):
        """
        When info cannot be serialized, we return None to avoid triggering error
        """
        return safe_json_encode(info)

    @field_serializer('feature', mode='plain')
    def serialize_feature(self, feature: Any, _feature):
        """
        We perform an automatic numpy serialization
        """
        if type(feature) == np.ndarray:
            return feature.tolist()
        elif type(feature).__module__ == np.__name__:
            # this is a Numpy integer or float, i.e., np.int32
            return feature.item()
        return feature


class Trajectory(list):
    def __init__(self, interactions: Union[List['InteractionBase'], None] = None) -> None:
        super().__init__(interactions or [])

    def __add__(self, other: Union['InteractionBase', 'Trajectory']):
        if isinstance(other, InteractionBase):
            return Trajectory(list(self) + [other])
        elif isinstance(other, Trajectory):
            return Trajectory(list(self) + list(other))
        else:
            raise ValueError(f"Unsupported type: {type(other)}")

    def __getstate__(self):
        return list(self)

    def __setstate__(self, state):
        super().__init__(state)

    def __repr__(self) -> str:
        return f"Trajectory({super().__repr__()})"

    def model_dump(self, **kwargs) -> List[Dict[str, Any]]:
        return [
            item.model_dump(**kwargs) if hasattr(item, 'model_dump')
            else item.__dict__
            for item in self
        ]

    def model_dump_json(self, **kwargs) -> str:
        return json.dumps(self.model_dump(**kwargs))
    
    @classmethod
    def model_validate(cls, data: List[Dict[str, Any]]) -> 'Trajectory':
        """
        Create a Trajectory from a list of dictionaries containing interaction data.
        Similar to Pydantic's model_validate method.
        """

        # we do a local import to avoid looped import
        from banditbench.tasks.mab.env import Interaction as MABInteraction, VerbalInteraction as MABVerbalInteraction
        from banditbench.tasks.cb.env import Interaction as CBInteraction, VerbalInteraction as CBVerbalInteraction

        interactions = []
        for item in data:
            # Determine interaction type from the data structure
            if 'mapped_action_name' in item:
                # Verbal interaction
                if 'state' in item:
                    # CB verbal interaction has state
                    interaction = CBVerbalInteraction.model_validate(item)
                else:
                    # MAB verbal interaction has no state
                    interaction = MABVerbalInteraction.model_validate(item)
            else:
                # Regular interaction
                if 'state' in item:
                    # CB interaction has state
                    interaction = CBInteraction.model_validate(item)
                else:
                    # MAB interaction has no state
                    interaction = MABInteraction.model_validate(item)
            interactions.append(interaction)
            
        return cls(interactions)


class InteractionBase:
    def __add__(self, other: Union['InteractionBase', 'Trajectory']) -> Trajectory:
        if isinstance(other, InteractionBase):
            return Trajectory(interactions=[self, other])
        elif isinstance(other, Trajectory):
            return Trajectory(interactions=[self] + other.interactions)
        else:
            raise ValueError(f"Unsupported type: {type(other)}")
