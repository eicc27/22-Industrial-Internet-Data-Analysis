import numpy as np


class Utils:
    def dim(data: np.ndarray):
        return len(data.shape)

    def filter_nnan(data: np.ndarray, ignore_features: list[str] = []):
        nan_idx = np.isnan(data)
        ignore_idx = np.delete(nan_idx, ignore_features, axis=1)
        row_idx = []
        col_idx = []
        for i, row in enumerate(ignore_idx):
            if True in row:
                continue
            row_idx.append(i)
        for i, col in enumerate(nan_idx.T):
            if True in col:
                continue
            col_idx.append(i)
        col_idx += ignore_features
        return row_idx, col_idx
    
    def filter_nan(data: np.ndarray, column_index: int):
        nan_idx = np.isnan(data[:, column_index])
        row_idx = []
        for i, row in enumerate(nan_idx):
            if row:
                row_idx.append(i)
        return row_idx

    def filter_by_rows(data: np.ndarray, rows: list[int]):
        return np.array([data[row] for row in rows])

    def filter_by_columns(data: np.ndarray, columns: list[int]):
        return np.array([data.T[column] for column in columns]).T
