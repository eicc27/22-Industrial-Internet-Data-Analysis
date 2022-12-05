import numpy as np
from sklearn import svm
from sklearn import preprocessing
from joblib import dump
import pandas as pd

def SVMClassifier(params):
    train_dataset = pd.read_csv("./src/train.csv")
    header = train_dataset.columns.values
    train_dataset = train_dataset.to_numpy()
    test_dataset = pd.read_csv("./src/test.csv")
    test_x = test_dataset.to_numpy()
    train_x = train_dataset[:, :-1]
    train_y = train_dataset[:, -1]
    enc = preprocessing.LabelEncoder()
    enc.fit(train_y)
    train_y = enc.transform(train_y)
    clf = svm.SVC()  # loss=params['loss'], penalty=params['penalty'], alpha=params['alpha'], max_iter=params['max_iter'], tol=params['tol'], shuffle=params['shuffle'], random_state=params['random_state']
    clf.fit(train_x, train_y)
    result = clf.predict(test_x)
    result = enc.inverse_transform(result)
    test_dataset[header[-1]] = result
    test_dataset.to_csv("./src/test_result.csv", index=False)
    dump(clf, "./src/train.model")
