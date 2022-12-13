import numpy as np
import hashlib
import time
import pandas as pd
import os

class Utils:
    def dim(data: np.ndarray):
        return len(data.shape)

    def filter_nnan_columns(data: np.ndarray, exception: int | None = None) -> np.ndarray:
        '''
        Filters data that does not contain nan value.

        Parameters
        -----
        data: the data to be filtered.
        exception: the column index that does not need to filter.
        '''
        _, columns = data.shape
        res = []
        for c in range(columns):
            if c == exception:
                res.append(True)
                continue
            col = data[:, c]
            isnan = False
            for i in col:
                if np.isnan(i):
                    res.append(False)
                    isnan = True
                    break
            if not isnan:
                res.append(True)
        return data[:, res], res

    def split_nan(data: np.ndarray, target: int):
        '''
        Splits the data, containing only one nan-filled column, into train and test dts.

        Parameters
        -----
        data: the data to be splitted.
        target: the target column index containing nan value.
        '''
        rows, _ = data.shape
        X = []
        y = []
        pred_data = []
        for r in range(rows):
            row = data[r]
            if np.isnan(row[target]): # append to pred_data
                pred_data.append(np.delete(row, target))
            else:
                X.append(np.delete(row, target))
                y.append(row[target])
        return np.array(X), np.array(y), np.array(pred_data)

    def strs_to_csv(fname: str):
        with open(fname, 'r') as f:
            _lines = f.readlines()
        with open(fname, 'w') as f:
            lines = [line[1:-2] + '\n' for line in _lines]
            f.writelines(lines)
        

    def gen_fname(fname: str):
        ext = fname.split('.')[-1]
        s = str(time.time()) + fname
        encrypted = hashlib.md5(bytes(s, encoding='utf-8')).hexdigest()
        return f'{encrypted[:8]}.{ext}'