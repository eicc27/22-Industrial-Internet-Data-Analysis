import numpy as np


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