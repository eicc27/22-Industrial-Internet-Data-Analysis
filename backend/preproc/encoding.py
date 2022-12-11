import numpy as np
from typing import Literal

class Encoding:
    '''
    Actual encoding is extremely complicated, and requires high-level NN to train for an apt representation(e.g. word2vec).
    This encoding method only supports simple one-hot and boolean representation for computational convenience.
    '''
    def __init__(self, data: np.ndarray, method: Literal['onehot', 'boolean']) -> None:
        self.data = data
        self.method = method
