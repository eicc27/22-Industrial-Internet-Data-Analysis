from sklearn.model_selection import train_test_split
import pandas as pd
import sys

try:
    data_csv = pd.read_csv(f'./backend/src/data.csv')
    labels = data_csv.columns
except:
    print("Cannot open file.")
    exit(1)
train_data, test_data = train_test_split(data_csv.to_numpy())
pd.DataFrame(train_data, columns=labels).to_csv('./train.csv')
pd.DataFrame(test_data, columns=labels).to_csv('./test.csv')
print('Splitting complete.')
