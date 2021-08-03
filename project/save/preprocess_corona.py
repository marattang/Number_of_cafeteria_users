import pandas as pd
import numpy as np

dataset = pd.read_excel('save/data/rawdata/corona.xlsx', usecols='A:B', header=5)
test = pd.read_csv('save/data/preprocess/prepro_test.csv')
train = pd.read_csv('save/data/preprocess/prepro_train.csv')

train_date = train['일자']
test_date = test['일자']

dataset.columns = ['일자', '확진자수']

dateset = pd.date_range('2016-02-01','2020-01-19')
dateset_df = pd.DataFrame([x for x in zip(dateset,np.repeat(0, 1449))])

dateset_df.columns = ['일자', '확진자수']
dataset = pd.concat([dateset_df, dataset], axis=0, ignore_index=True)

train_mask = dataset['일자'].isin(train_date)
test_mask = dataset['일자'].isin(test_date)

train = dataset.loc[train_mask, :]
test = dataset.loc[test_mask, :]

# train.to_csv('save/data/preprocess/corona_train.csv', index=False)
train.to_csv('save/data/preprocess/corona_train_2016.csv', index=False)
test.to_csv('save/data/preprocess/corona_test.csv', index=False)
