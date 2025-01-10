import numpy as np
from banditbench.tasks.cb import State

def test_save_state():
    print()
    s = State(feature=[1, 2, 3], index=1, info=None)
    print(s.model_dump_json())
    s = State(feature=np.array([1, 2, 3]), index=1, info={'field1': 1})
    print(s.model_dump_json())
    s = State(feature=np.int32(3), index=1, info={'field1': np.array([1,2,3])})
    print(s.model_dump_json())

test_save_state()