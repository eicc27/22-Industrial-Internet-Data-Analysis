import numpy as np
from typing import Literal
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score, ShuffleSplit
from sklearn.feature_selection import f_regression
import matplotlib.pyplot as plt

from preproc.logger import Logger


class FeatureSelection:
    '''
    Instead of directly wapping out a feature, this class returns the corelation table to leave the 
    right to users.

    Supports
    -----
    pearson: Use Pearsion Correlation to determine the relativity of each feature. This 
    method only calculates LINEAR RELATIVITY that ranges from -1 to 1.
    forest: Use Random Forest Regressor to determine the cross relativity score of each feature.
    '''

    def __init__(self, data: np.ndarray, labels: np.ndarray, target: np.ndarray, method: Literal['forest', 'pearson']) -> None:
        self.data = data
        self.labels = labels
        self.target = target
        self.method = method

    def run(self):
        '''
        FS job dispatcher.
        '''
        match self.method:
            case 'forest':
                return self._forest_fs()
            case 'pearson':
                return self._pearson_fs()
            case _:
                Logger(f'Unknown method: {self.method}').log('error')
                raise ValueError()

    def _pearson_fs(self):
        pv = f_regression(self.data, self.target)[1]
        retval = []
        for label, value in zip(self.labels, pv):
            retval.append((label, value))
        return sorted(retval, key=lambda k: k[1], reverse=True)

    def _forest_fs(self):
        fr = RandomForestRegressor(n_estimators=30, max_depth=5)
        scores = []
        _, columns = self.data.shape
        for col in range(columns):
            column = self.data[:, col]
            score = cross_val_score(fr, column.reshape(-1, 1), self.target, scoring='r2', cv=ShuffleSplit())
            score = np.mean(score)
            scores.append(score)
        retval = []
        for label, score in zip(self.labels, scores):
            retval.append((label, score))
        return sorted(retval, key=lambda k: k[1], reverse=True)

    def savefig(self):
        result = self.run()
        print(result)
        plt.clf()
        for r in result:
            plt.barh(r[0], r[1], color='orange')
        plt.title('Correlation value of labels')
        plt.savefig('./src/fs.png')