import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

train = pd.read_csv('./save/data/preprocess/prepro_train.csv')
test = pd.read_csv('./save/data/preprocess/prepro_test.csv')

train['요일'] = train['요일'].map({'월':0, '화':1, '수':2, '목':3, '금':4})
test['요일'] = test['요일'].map({'월':0, '화':1, '수':2, '목':3, '금':4})

x_train = train[['요일', '본사정원수', '본사출장자수', '본사시간외근무명령서승인건수', '현본사소속재택근무자수']]
y1_train = train['중식계']
y2_train = train['석식계']

x_test = test[['요일', '본사정원수', '본사출장자수', '본사시간외근무명령서승인건수', '현본사소속재택근무자수']]
y1_test = test['중식계']
y2_test = test['석식계']

model1 = RandomForestRegressor(n_jobs=-1, random_state=10)
model2 = RandomForestRegressor(n_jobs=-1, random_state=10)

model1.fit(x_train, y1_train)
model2.fit(x_train, y2_train)

pred1 = model1.predict(x_test)
pred2 = model2.predict(x_test)

r2 = r2_score([y1_test, y2_test],[pred1, pred2])
print('r2 score : ', r2)