import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.ensemble import IsolationForest
from typing import Literal

from preproc.logger import Logger


class Sifting:
    '''
    Specifies and executes a given sifting method.
    *Requires a big-enough DS to operate if using ML methods.*

    Supports
    -----
    MAD: Median Absolute Deviation, median(|xi - median|), must be given a threshold as VALUE. ⭐⭐⭐
    DBSCAN: Density-Based Spatial Clustering of Applications with Noise, a PLASTIC, AUTOMATIC clustering algorithm.
    Must be given a threshold as the maximum DISTANCE to identify a cluster. ⭐⭐⭐⭐
    if: Isolation Forest, CAN be given a threshold as PERCENTAGE of anomaly(leaving None means auto). ⭐⭐⭐⭐⭐
    '''

    def __init__(self, data: np.ndarray, method: Literal['mad', 'dbscan', 'if'], column: int, threshold: float | None = None) -> None:
        self.data = data.astype(np.float32)
        self.column = self.data[:, column]
        self.threshold = threshold
        self.method = method

    def run(self):
        '''
        Job dispatcher for different sifting methods.
        '''
        match self.method:
            # case 'mad':
            #     return self._mad_sifting()
            case 'dbscan':
                return self._dbscan_sifting()
            case 'if':
                return self._if_sifting()
            case _:
                Logger(f'Invalid sifting method specified: {self.method}.').log(
                    'error')
                raise ValueError 

    def _mad_sifting(self):
        mad_column = np.abs(self.column - np.median(self.column))
        sifting_row = mad_column < self.threshold
        self.data = self.data[sifting_row]
        return self.data

    def _dbscan_sifting(self):
        dbscan = DBSCAN(eps=self.threshold).fit(self.data[:, :-1])
        sifting_row = dbscan.labels_ > 0
        self.data = self.data[sifting_row]
        return self.data

    def _if_sifting(self):
        forest = IsolationForest(
            contamination=self.threshold if self.threshold else 'auto')
        sifting_row = forest.fit_predict(self.data[:, :-1]) > 0
        self.data = self.data[sifting_row]
        return self.data
