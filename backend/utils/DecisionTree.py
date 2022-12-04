import numpy as np
from sklearn import tree
import pandas as pd

def DTClassifier(params):
    train_dataset = pd.read_csv("./src/train.csv").to_numpy()
    test_dataset = pd.read_csv("./src/test.csv").to_numpy()
    print(params)
    print(train_dataset, test_dataset)

