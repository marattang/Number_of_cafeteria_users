import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor

train = pd.read_csv('./save/data/rawdata/train.csv')
test = pd.read_csv('./save/data/rawdata/test.csv')
submission = pd.read_csv('./save/data/rawdata/sample_submission.csv')

train.head()
test.head()
# submission.head()

train['요일'] = train['요일'].map({'월':0, '화':1, '수':2, '목':3, '금':4})
test['요일'] = test['요일'].map({'월':0, '화':1, '수':2, '목':3, '금':4})

x_train = train[['요일', '본사정원수', '본사출장자수', '본사시간외근무명령서승인건수', '현본사소속재택근무자수']]
y1_train = train['중식계']
y2_train = train['석식계']

x_test = test[['요일', '본사정원수', '본사출장자수', '본사시간외근무명령서승인건수', '현본사소속재택근무자수']]

model1 = RandomForestRegressor(n_jobs=-1, random_state=42)
model2 = RandomForestRegressor(n_jobs=-1, random_state=42)

model1.fit(x_train, y1_train)
model2.fit(x_train, y2_train)

pred1 = model1.predict(x_test)
pred2 = model2.predict(x_test)

submission['중식계'] = pred1
submission['석식계'] = pred2

submission.to_csv('./save/data/submission.csv', index=False)