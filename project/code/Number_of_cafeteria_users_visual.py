import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt

train = pd.read_csv('./save/data/preprocess/prepro_train.csv')
test = pd.read_csv('./save/data/preprocess/prepro_test.csv')

date_train_lunch = pd.read_csv('./save/data/preprocess/date_train_lunch.csv')
corona = pd.read_csv('./save/data/preprocess/corona_train.csv')
date_train_dinner = pd.read_csv('./save/data/preprocess/date_train_dinner.csv')

# 
print(train.head())
print(date_train_lunch.head())

train['일자'] = pd.to_datetime(train['일자'])
train_mask = (train['일자'] >= '2020-08-15') & (train['일자'] <= '2020-09-15')
train = train.loc[train_mask, :]
date = train['일자']
cnt = train['본사정원수'] - train['본사휴가자수'] - train['본사출장자수'] - train['본사시간외근무명령서승인건수'] - train['현본사소속재택근무자수']
lunch = train['중식계']
dinner = train['석식계']

mask1 = (date_train_lunch['날짜'] >= '2020-08-15') & (date_train_lunch['날짜'] <= '2020-09-15')
mask2 = (date_train_dinner['날짜'] >= '2020-08-15') & (date_train_dinner['날짜'] <= '2020-09-15')
corona_mask = (corona['일자'] >= '2020-08-15') & (corona['일자'] <= '2020-09-15')

date_train_lunch = date_train_lunch.loc[mask1, :]
date_train_dinner = date_train_dinner.loc[mask2, :]
corona = corona.loc[corona_mask, :]

rain = date_train_lunch['강수량(mm)']
temper = date_train_lunch['기온(°C)']
snow = date_train_lunch['적설(cm)']
corona_cnt = corona['확진자수']

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.figure(figsize=(10,5))

# plt.plot(date, temper)
# plt.title('2019년 7월 기온(°C)', fontsize=15)
# plt.xticks(rotation=90)

plt.plot(date, (lunch/cnt)*100)
plt.title('2020년 2차 대유행 시기 중식인원비', fontsize=15)
plt.xticks(rotation=90)

# plt.plot(date, rain)
# plt.title('2019년 9월 ~ 10월 강수량(mm)', fontsize=15)
# plt.xticks(rotation=90)

# plt.plot(date, corona_cnt)
# plt.title('2020년 2차 대유행 시기 코로나 확진자 수', fontsize=15)
# plt.xticks(rotation=90)

plt.show()