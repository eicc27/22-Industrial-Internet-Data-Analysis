import numpy as np
from sklearn import linear_model
from sklearn.metrics import mean_absolute_percentage_error
from joblib import dump, load
import pandas as pd

def LinearRegression(loss, max_iter, shuffle, random_state, tol, penalty, alpha):
    train_dataset = pd.read_csv("./src/train.csv").to_numpy()
    test_dataset = pd.read_csv("./src/test.csv").to_numpy()
    test_x = test_dataset[:, :-1]
    test_y = test_dataset[:, -1]
    train_x = train_dataset[:, :-1]
    train_y = train_dataset[:, -1]
    clf = linear_model.SGDRegressor(loss=loss, penalty=penalty, alpha=alpha, max_iter=max_iter, tol=tol, shuffle=shuffle, random_state=random_state)
    clf.fit(train_x, train_y)
    y_pred = clf.predict(test_x)
    dump(clf, "./src/model.model")
    print(y_pred)
    print(mean_absolute_percentage_error(test_y, y_pred))
    return mean_absolute_percentage_error(test_y, y_pred)


def LinearRegressionPredict():
    clf = load("./src/model.model")
    predict_dataset = pd.read_csv("./src/predict.csv")
    predict_x = predict_dataset.to_numpy()
    result = clf.predict(predict_x)
    predict_dataset['result'] = np.around(result, 2)
    predict_dataset.to_csv('./src/result.csv', index=False)
