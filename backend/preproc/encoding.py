import numpy as np
from typing import Literal

class Encoding:
    def __init__(self, data: np.ndarray, method: Literal['onehot', 'boolean']) -> None:
        self.data = data
        self.method = method
