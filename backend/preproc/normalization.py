import numpy as np
from typing import Literal

from logger import Logger

NormMethods = Literal['minmax', 'zscore', 'sigmoid', 'log', 'l2']

class Norm:
    '''
    Specifies and executes a given normalization method.

    Supports
    -----
    minmax: LINEAR, (x - min) / (max - min) ⭐⭐⭐⭐
    z-score: LINEAR, (x - mu) / (sigma) ⭐⭐⭐⭐⭐
    sigmoid: NONLINEAR, "S"-shape, has great depression on discursive data, 1 / (1 + e^-x), ⭐⭐⭐
    log: NONLINEAR, has great leveling effect on big data, log(x) / log(max), ⭐⭐
    l2: LINEAR, REQUIRES MULTIDIM DATA(vector), good at representing vectors(esp. in NLP, e.g. word vec), x / sqrt(sigma(xi^2)), ⭐
    '''

    def __init__(self, data: np.ndarray, column_index: int, method: NormMethods) -> None:
        self.data = data.astype(np.float32)
        self.column_index = column_index
        self.column = self.data[:, column_index]
        self.method: NormMethods = method

    def run(self):
        '''
        Job dispatcher for different normalization methods.
        '''
        match self.method:
            case 'minmax':
                return self._minmax_norm()
            case 'zscore':
                return self._zscore_norm()
            case 'sigmoid':
                return self._sigmoid_norm()
            case 'log':
                return self._log_norm()
            case 'l2':
                return self._l2_norm()
            case _: # internal error while passing params
                Logger(f'Invalid norm method specified: {self.method}.').log('error')
                exit(1)

    def _minmax_norm(self):
        maxval = np.max(self.column)
        minval = np.min(self.column)
        self.data[:, self.column_index] = (self.column - minval) / (maxval - minval)
        return self.data

    def _zscore_norm(self):
        mu = np.average(self.column)
        sigma = np.std(self.column)
        self.data[:, self.column_index] = (self.column - mu) / sigma
        return self.data

    def _sigmoid_norm(self):
        self.data[:, self.column_index] = 1 / (1 + np.exp(-self.column))
        return self.data
    
    def _log_norm(self):
        maxval = np.max(self.column)
        self.data[:, self.column_index] = np.log10(self.column) / np.log10(maxval)
        return self.data

    def _l2_norm(self):
        self.data[:, self.column_index] = self.column / np.linalg.norm(self.column)
    