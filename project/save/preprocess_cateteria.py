import pandas as pd
import numpy as np

dataset = pd.read_csv('save/data/rawdata/train.csv')

train_mask = dataset['일자'] <= '2020-11-01'
test_mask = dataset['일자'] >= '2020-11-01'

train = dataset.loc[train_mask, :]
test = dataset.loc[test_mask, :]

train.to_csv('save/data/preprocess/prepro_train.csv')
test.to_csv('save/data/preprocess/prepro_test.csv')