import numpy as np
from typing import Literal

'''
Specifies and executes a given padding method.
'''
class Padding:
    def __init__(self, data: np.ndarray, method: Literal['zero', 'mean', 'ml', 'mi', 'clustermean', 'bayesian']) -> None:
        self.data = data
        self.method = method
    
    def zero_padding(self):
        pass