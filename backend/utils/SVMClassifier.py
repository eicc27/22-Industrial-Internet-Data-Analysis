import numpy as np
from sklearn import svm
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from joblib import dump, load
import pandas as pd

def SVMClassifier(C, kernel, degree, tol, max_iter, random_state):
    train_dataset = pd.read_csv("./src/train.csv").to_numpy()
    test_dataset = pd.read_csv("./src/test.csv").to_numpy()
    test_x = test_dataset[:, :-1]
    test_y = test_dataset[:, -1]
    train_x = train_dataset[:, :-1]
    train_y = train_dataset[:, -1]
    enc = preprocessing.LabelEncoder()
    enc.fit(train_y)
    train_y = enc.transform(train_y)
    clf = svm.SVC(C=C, kernel=kernel, degree=degree, tol=tol, max_iter=max_iter, random_state=random_state)
    clf.fit(train_x, train_y)
    y_pred = clf.predict(test_x)
    dump(enc, "./src/encoder.model")
    dump(clf, "./src/model.model")
    test_y = enc.transform(test_y)
    return accuracy_score(test_y, y_pred)


def SVMClassifierPredict():
    clf = load("./src/model.model")
    enc = load("./src/encoder.model")
    predict_dataset = pd.read_csv("./src/predict.csv")
    predict_x = predict_dataset.to_numpy()
    result = clf.predict(predict_x)
    predict_dataset['result'] = enc.inverse_transform(result)
    predict_dataset.to_csv('./src/result.csv', index=False)
