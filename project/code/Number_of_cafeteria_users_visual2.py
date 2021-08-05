import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

original = pd.read_csv('./save/data/preprocess/test_result_original.csv')
predict_RF = pd.read_csv('./save/data/preprocess/test_result_predict.csv')
predict_RF_onebon = pd.read_csv('./save/data/preprocess/test_result_predict_RF_onebon.csv')
predict_RNN = pd.read_csv('./save/data/preprocess/test_result_predict_RNN.csv')
predict_DNN = pd.read_csv('./save/data/preprocess/test_result_predict_DNN.csv')
predict_CNN = pd.read_csv('./save/data/preprocess/test_result_predict_CNN.csv')

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.figure(figsize=(10,5))

# plt.plot(date, temper)
# plt.title('2019년 7월 기온(°C)', fontsize=15)
# plt.xticks(rotation=90)

print(original)
original['일자'] = pd.to_datetime(original['일자'])
date = original['일자']
lunch = original.loc[:, '중식계']
RNN_lunch = predict_RNN.iloc[:, 2]
RF_lunch = predict_RF.iloc[:, 2]
CNN_lunch = predict_CNN.iloc[:, 2]
DNN_lunch = predict_DNN.iloc[:, 2]
RF_lunch_onebon = predict_RF_onebon.iloc[:, 2]
print(RNN_lunch)
'''
fig,ax=plt.subplots(nrows=2,ncols=2)

ax[0,0].plot(date, lunch, date, RNN_lunch)
ax[0,1].plot(date, lunch, date, RF_lunch)
ax[1,0].plot(date, lunch, date, CNN_lunch)
ax[1,1].plot(date, lunch, date, DNN_lunch)
'''
'''
plt.subplot(2, 2, 1)
plt.plot(date, lunch, label='original')
plt.plot(date, RNN_lunch, label='RNN')
plt.title('RNN')
plt.legend()

plt.subplot(2, 2, 2)
plt.plot(date, lunch, label='original')
plt.plot(date, RF_lunch, label='Random Forest')
plt.title('Random Forest')
plt.legend()

plt.subplot(2, 2, 3)
plt.plot(date, lunch, label='original')
plt.plot(date, CNN_lunch, label='CNN')
plt.title('CNN')
plt.legend()

plt.subplot(2, 2, 4)
plt.plot(date, lunch, label='original')
plt.plot(date, DNN_lunch, label='DNN')
plt.title('DNN')
plt.legend()
'''
plt.subplot(2, 1, 1)
plt.plot(date, lunch, label='original')
plt.plot(date, RF_lunch_onebon, label='original')
plt.title('original')
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(date, lunch, label='original')
plt.plot(date, RF_lunch, label='external data')
plt.title('external data')
plt.legend()
# plt.title('2020년 10월 ~ 2021년 1월 중식계', fontsize=15)
# plt.xticks(rotation=90)
plt.show()