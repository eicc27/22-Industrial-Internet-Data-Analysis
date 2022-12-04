import numpy as np
from sklearn import linear_model
import pandas as pd

def LinearRegression(params):
    train_dataset = pd.read_csv("./src/train.csv")
    header = train_dataset.columns.values
    train_dataset = train_dataset.to_numpy()
    test_dataset = pd.read_csv("./src/test.csv")
    test_x = test_dataset.to_numpy()
    train_x = train_dataset[:, :-1]
    train_y = train_dataset[:, -1]
    clf = linear_model.SGDRegressor(loss=params['loss'], penalty=params['penalty'], alpha=params['alpha'], max_iter=params['max_iter'], tol=params['tol'], shuffle=params['shuffle'], random_state=params['random_state'])
    clf.fit(train_x, train_y)
    result = clf.predict(test_x)
    test_dataset[header[-1]] = np.around(result, 2)
    test_dataset.to_csv("./src/test_result.csv")



