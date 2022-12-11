import numpy as np
from typing import Literal
from utilities import Utils
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.base import ClassifierMixin, RegressorMixin
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from logger import Logger

PaddingMethods = Literal['zero', 'mean', 'median',
                         'knnc', 'knnr', 'forestc', 'forestr']


class Padding:
    '''
    Specifies and executes a given padding method.

    Supports
    -----
    zero: zero padding
    mean: mean padding
    median: median padding
    knn: KNN classifier(knnc) & regressor(knnr)
    forest: Random Forest classifier(forestc) & regressor(forestr)
    '''

    def __init__(self, data: np.ndarray, column_index: int, method: PaddingMethods) -> None:
        self.data = data.astype(np.float32)
        self.column_index = column_index
        self.column = self.data[:, column_index]
        self.method: PaddingMethods = method

    def run(self):
        '''
        Job dispatcher for different prediction methods.
        '''
        match self.method:
            case 'zero':
                return self._zero_padding()
            case 'mean':
                return self._mean_padding()
            case 'median':
                return self._median_padding()
            case 'knnc':
                try:
                    return self._knnc_padding()
                except ValueError:  # the value is not suitable for classification
                    Logger(
                        f"The column(idx. {self.column_index}) is not suitable for classification. Turned to KNN-Regressor instead.").log('warn')
                    return self._knnr_padding()
            case 'knnr':
                return self._knnr_padding()
            case 'forestc':
                try:
                    return self._forestc_padding()
                except ValueError:  # the value is not suitable for classification
                    Logger(
                        f"The column(idx. {self.column_index}) is not suitable for classification. Turned to RF-Regressor instead.").log('warn')
                    return self._knnr_padding()
            case 'forestr':
                return self._forestr_padding()
            case '_':  # internal error while passing params
                Logger(f'Invalid padding method specified: {self.method}.').log(
                    'error')
                exit(1)

    def _zero_padding(self):
        self.column[np.isnan(self.column)] = 0
        return self.data

    def _mean_padding(self):
        self.column[np.isnan(self.column)] = self.column.mean
        return self.data

    def _median_padding(self):
        self.column[np.isnan(self.column)] = np.median(self.column)
        return self.data

    def _knnc_padding(self):
        return self._iterative_padding(KNeighborsClassifier)

    def _knnr_padding(self):
        return self._iterative_padding(KNeighborsRegressor)

    def _forestc_padding(self):
        return self._iterative_padding(RandomForestClassifier)

    def _forestr_padding(self):
        return self._iterative_padding(RandomForestRegressor)

    def _iterative_padding(self, method: ClassifierMixin | RegressorMixin):
        '''
        Extract the intersection of not-nan columns as X, the given column as y, 
        and run the given iterative ML prediction algorithm.
        '''
        data, nnans = Utils.filter_nnan_columns(
            self.data, exception=self.column_index)
        if not data.shape[1]:  # array has no not-nan columns
            Logger("Every data column contains nan. Use traditional padding method first.").log(
                'error')
            return 1
        target_index = 0
        for i, nnan in enumerate(nnans):
            if i >= self.column_index:
                break
            if nnan:
                target_index += 1
        X, y, pred_data = Utils.split_nan(data, target_index)
        if not pred_data.shape[0]:  # array has no nan columns to predict
            Logger(f"Every data in this column(idx. {self.column_index}) has a value. Skipped.").log(
                'warn')
            return self.data
        model = method().fit(X, y)
        predictions = model.predict(pred_data)
        # fills in the blank
        rows, _ = self.data.shape
        index = 0
        for r in range(rows):
            if np.isnan(self.data[r, self.column_index]):
                self.data[r, self.column_index] = predictions[index]
                index += 1
        return self.data
