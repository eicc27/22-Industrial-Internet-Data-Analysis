import numpy as np
from typing import Literal

class Norm:
    def __init__(self, data: np.ndarray, method: Literal['minmax', 'zscore', ]) -> None:
        self.data = data
        self.method = method