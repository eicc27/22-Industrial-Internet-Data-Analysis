import numpy as np
from typing import Literal
from utils import Utils
from sklearn.neighbors import KNeighborsClassifier


'''
Specifies and executes a given padding method.
'''
class Padding:
    def __init__(self, data: np.ndarray, column_index: int, method: Literal['zero', 'mean', 'median', 'knn', 'forestc', 'forestr']) -> None:
        self.data = data
        self.column = self.data[:, column_index]
        self.method = method
    
    def run(self):
        pass
    
    def zero_padding(self):
        self.column[np.isnan(self.column)] = 0
        return self.data
    
    def mean_padding(self):
        self.column[np.isnan(self.column)] = self.column.mean
        return self.data
    
    def median_padding(self):
        self.column[np.isnan(self.column)] = np.median(self.column)
        return self.data
    
    def knn_padding(self):
        rows, cols = Utils.filter_nnan(self.data)
        # cols.append(3) # predict column 3
        train_data = Utils.filter_by_rows(self.data, rows)
        train_data = Utils.filter_by_columns(train_data, cols)
        y = self.data[:, self.column]
        print(train_data.shape)
        km = KNeighborsClassifier().fit(train_data, Utils.filter_by_rows(y, rows))
        prows, _ = Utils.filter_nnan(self.data, ignore_features=[self.column])
        pred_data = Utils.filter_by_rows(self.data, prows)
        pred_data = pred_data[np.isnan(pred_data[:, self.column])]
        pred_data = Utils.filter_by_columns(pred_data, cols)
        pred = km.predict(pred_data)