import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

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


# scaler = QuantileTransformer()
# x_train = scaler.fit_transform(x_train['본사정원수', '본사출장자수', '본사시간외근무명령서승인건수', '현본사소속재택근무자수'])
# x_train = scaler.fit_transform(x_train)
# x_test = scaler.fit(x_test['본사정원수', '본사출장자수', '본사시간외근무명령서승인건수', '현본사소속재택근무자수'])
# x_test = scaler.fit(x_test)

# lunch_train = date_train_lunch[['기온(°C)','강수량(mm)','습도(%)','적설(cm)']]
# dinner_train = date_train_dinner[['기온(°C)','강수량(mm)','습도(%)','적설(cm)']]

lunch_train = date_train_lunch[['강수량(mm)']]
dinner_train = date_train_dinner[['강수량(mm)']]

corona_train = corona_train[['확진자수']]
corona_test = corona_test[['확진자수']]

x1_train = pd.concat([x_train, lunch_train, corona_train], axis=1)
x2_train = pd.concat([x_train, dinner_train, corona_train], axis=1)

# 

# lunch_test = date_test_lunch[['기온(°C)','강수량(mm)','습도(%)','적설(cm)']]
# dinner_test = date_test_dinner[['기온(°C)','강수량(mm)','습도(%)','적설(cm)']]
lunch_test = date_test_lunch[['강수량(mm)']]
dinner_test = date_test_dinner[['강수량(mm)']]

x1_test = pd.concat([x_test, lunch_test, corona_test], axis=1)
x2_test = pd.concat([x_test, dinner_test, corona_test], axis=1)

model1 = RandomForestRegressor(n_jobs=-1, random_state=5)
model2 = RandomForestRegressor(n_jobs=-1, random_state=5)

print(lunch_train)

model1.fit(x1_train, y1_train)
model2.fit(x2_train, y2_train)

pred1 = model1.predict(x1_test)
pred2 = model2.predict(x2_test)

r2 = r2_score([y1_test, y2_test],[pred1, pred2])
print('r2 score : ', r2)

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