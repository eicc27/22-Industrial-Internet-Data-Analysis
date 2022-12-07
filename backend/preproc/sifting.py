import numpy as np
from typing import Literal

class Sifting:
    def __init__(self, data: np.ndarray, method: Literal['mad', '3sig', 'cluster']) -> None:
        self.data = data
        self.method = method