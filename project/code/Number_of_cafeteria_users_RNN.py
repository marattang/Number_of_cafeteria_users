import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from tensorflow.keras.layers import Dense, LSTM, Flatten, Input, SimpleRNN
from tensorflow.keras.models import Model, Sequential
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler, RobustScaler, StandardScaler, QuantileTransformer, PowerTransformer
from tensorflow.python.keras.callbacks import EarlyStopping
from tensorflow.python.keras.layers.core import Dropout

train = pd.read_csv('./save/data/preprocess/prepro_train.csv')
test = pd.read_csv('./save/data/preprocess/prepro_test.csv')

date_train_lunch = pd.read_csv('./save/data/preprocess/date_train_lunch.csv')
date_train_dinner = pd.read_csv('./save/data/preprocess/date_train_dinner.csv')
date_test_lunch = pd.read_csv('./save/data/preprocess/date_test_lunch.csv')
date_test_dinner = pd.read_csv('./save/data/preprocess/date_test_dinner.csv')

corona_train = pd.read_csv('./save/data/preprocess/corona_train_2016.csv')
corona_test = pd.read_csv('./save/data/preprocess/corona_test.csv')

train['요일'] = train['요일'].map({'월':0, '화':1, '수':2, '목':3, '금':4})
test['요일'] = test['요일'].map({'월':0, '화':1, '수':2, '목':3, '금':4})

x_train = train[['요일', '본사정원수', '본사휴가자수', '본사출장자수', '본사시간외근무명령서승인건수', '현본사소속재택근무자수']]
y1_train = train['중식계']
y2_train = train['석식계']

x_test = test[['요일', '본사정원수', '본사휴가자수', '본사출장자수', '본사시간외근무명령서승인건수', '현본사소속재택근무자수']]
y1_test = test['중식계']
y2_test = test['석식계']

# lunch_train = date_train_lunch[['기온(°C)','강수량(mm)','습도(%)','적설(cm)']]
# dinner_train = date_train_dinner[['기온(°C)','강수량(mm)','습도(%)','적설(cm)']]

lunch_train = date_train_lunch[['강수량(mm)']]
dinner_train = date_train_dinner[['강수량(mm)']]

corona_train = corona_train[['확진자수']]
corona_test = corona_test[['확진자수']]

x1_train = pd.concat([x_train, lunch_train, corona_train], axis=1)
x2_train = pd.concat([x_train, dinner_train, corona_train], axis=1)

# lunch_test = date_test_lunch[['기온(°C)','강수량(mm)','습도(%)','적설(cm)']]
# dinner_test = date_test_dinner[['기온(°C)','강수량(mm)','습도(%)','적설(cm)']]
lunch_test = date_test_lunch[['강수량(mm)']]
dinner_test = date_test_dinner[['강수량(mm)']]

x1_test = pd.concat([x_test, lunch_test, corona_test], axis=1)
x2_test = pd.concat([x_test, dinner_test, corona_test], axis=1)

print('x train', x1_train)
print('x test', x1_test)
# model
# [1149 rows x 8 columns]
x1_train = x1_train.to_numpy()
x1_test = x1_test.to_numpy()
x2_train = x2_train.to_numpy()
x2_test = x2_test.to_numpy()
# 56 8
print('x_train', x1_train.shape)

scaler = QuantileTransformer()
x1_train = scaler.fit_transform(x1_train)
x1_test = scaler.transform(x1_test)
x2_train = scaler.transform(x2_train)
x2_test = scaler.transform(x2_test)

x1_train = x1_train.reshape(1149, 8, 1)
x2_train = x2_train.reshape(1149, 8, 1)
x1_test = x1_test.reshape(56, 8, 1)
x2_test = x2_test.reshape(56, 8, 1)
# x1_train, y1_train,
print('x1_train shape', x1_train.shape)
print('y1_train shape', y1_train.shape)

model1 = Sequential()
model1.add(LSTM(units=64, activation='relu', input_shape=(8, 1)))
# model1.add(Dropout(0.4))
model1.add(Dense(64, activation='relu'))
model1.add(Dense(32, activation='relu'))
model1.add(Dense(8, activation='relu'))
# model1.add(Dropout(0.2))
model1.add(Dense(1))

model2 = Sequential()
model2.add(LSTM(units=64, activation='relu', input_shape=(8, 1)))
# model2.add(Dropout(0.4))
model2.add(Dense(128, activation='relu'))
model2.add(Dense(64, activation='relu'))
model2.add(Dense(32, activation='relu'))
model2.add(Dense(8, activation='relu'))
# model2.add(Dropout(0.2))
model2.add(Dense(1))

model1.compile(loss='mse', optimizer='adam')
model2.compile(loss='mse', optimizer='adam')

es = EarlyStopping(monitor='val_loss', mode='auto', patience=5)

model1.fit(x1_train, y1_train, validation_split=0.1, batch_size=5, callbacks=[es], epochs=80)
model2.fit(x2_train, y2_train, validation_split=0.1, batch_size=5, callbacks=[es], epochs=80)

pred1 = model1.predict(x1_test)
pred2 = model2.predict(x2_test)

print('pred1.shape',pred1.shape)
print('pred2.shape',pred2.shape)
print('y1_test.shape',y1_test.shape)
print('y2_test.shape',y2_test.shape)
pred1 = pred1.reshape(pred1.shape[0])
pred2 = pred2.reshape(pred2.shape[0])

r2 = r2_score([y1_test, y2_test],[pred1, pred2])
print('r2 score : ', r2)

# 랜덤 포레스트
# 기온(°C)  강수량(mm)  습도(%)  적설(cm)
# r2 score :  0.7549222139625361 -> r2 score :  0.7715592633957508

#  강수량(mm)
# r2 score :  0.8024686160736393 -> r2 score :  0.809803745030216 미미하게 증가

# 원본
# r2 score :  0.8043391132655167 -> r2 score :  0.809094699518922

# 본사 휴가자수 컬럼 포함
# r2 score :  0.8118143854009736

# 코로나 변수 추가
# r2 score :  0.8223234403614652

# RNN 배운걸로만 단순 구현했을 때
# SimpleRNN r2 score : 0.24451711503429635
# LSTM = r2 score :  0.5280218990011818