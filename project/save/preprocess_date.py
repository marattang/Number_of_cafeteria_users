import pandas as pd
import numpy as np
from datetime import datetime
import time

# 기상청 날씨 데이터 전처리
data2016 = pd.read_csv('save/data/rawdata/기상청/2016_01_to_12.csv', encoding='euc-kr')
data2017 = pd.read_csv('save/data/rawdata/기상청/2017_01_to_12.csv', encoding='euc-kr')
data2018 = pd.read_csv('save/data/rawdata/기상청/2018_01_to_12.csv', encoding='euc-kr')
data2019 = pd.read_csv('save/data/rawdata/기상청/2019_01_to_12.csv', encoding='euc-kr')
data2020 = pd.read_csv('save/data/rawdata/기상청/2020_01_to_12.csv', encoding='euc-kr')
data2021 = pd.read_csv('save/data/rawdata/기상청/2021_01_to_07.csv', encoding='euc-kr')
train = pd.read_csv('save/data/rawdata/train.csv')
merge_df = pd.concat([data2016, data2017, data2018, data2019, data2020, data2021])

# 필요 없는 컬럼 제거
merge_df.drop(['지점', '지점명', '기온 QC플래그', '강수량 QC플래그',  '풍속 QC플래그', '풍향(16방위)',
               '풍향 QC플래그', '습도 QC플래그', '증기압(hPa)', '이슬점온도(°C)',  '3시간신적설(cm)'], axis=1, inplace=True)

# 중식, 석식 시간대 데이터 필터링
mask_d = merge_df['일시'].str.slice(start=10).str.strip() =='18:00'
mask_l = merge_df['일시'].str.slice(start=10).str.strip() == '12:00'

merge_df_d = merge_df.loc[mask_d,:]
merge_df_l = merge_df.loc[mask_l,:]

# 날짜 뽑아내기
merge_df_l['날짜'] = merge_df_l.iloc[:, 0].str[0:10]
merge_df_d['날짜'] = merge_df_d.iloc[:, 0].str[0:10]

# 날짜형으로 형변환
merge_df_l['날짜'] = pd.to_datetime(merge_df_l.iloc[:,6])
merge_df_d['날짜'] = pd.to_datetime(merge_df_d.iloc[:,6])

# 요일 추가
merge_df_l['요일'] = merge_df_l['날짜'].dt.dayofweek
merge_df_d['요일'] = merge_df_d['날짜'].dt.dayofweek

# 결측치 제거
merge_df_l = merge_df_l.fillna(0)
merge_df_d = merge_df_d.fillna(0)

# 평일만 필터링
mask_l = (merge_df_l['요일'] != 5) & (merge_df_l['요일'] != 6)
mask_d = (merge_df_d['요일'] != 5) & (merge_df_d['요일'] != 6)

merge_df_l = merge_df_l.loc[mask_l, :]
merge_df_d = merge_df_d.loc[mask_d, :]

# 요일 매핑
# merge_df_l['요일'] = merge_df_l['요일'].map({0:'월', 1:'화',2:'수',3:'목',4:'금'})
# merge_df_d['요일'] = merge_df_d['요일'].map({0:'월', 1:'화',2:'수',3:'목',4:'금'})

# 날짜 필터링
train_date = train['일자']
merge_df_l_mask = merge_df_l['날짜'].isin(train_date)
merge_df_l = merge_df_l.loc[merge_df_l_mask, :]

merge_df_d_mask = merge_df_d['날짜'].isin(train_date)
merge_df_d = merge_df_d.loc[merge_df_d_mask, :]

date_train_d_mask = merge_df_d['날짜'] <= '2020-10-31'
date_test_d_mask = merge_df_d['날짜'] > '2020-10-31'
date_train_l_mask = merge_df_l['날짜'] <= '2020-10-31'
date_test_l_mask = merge_df_l['날짜'] > '2020-10-31'

date_train_d = merge_df_d.loc[date_train_d_mask, :]
date_test_d = merge_df_d.loc[date_test_d_mask, :]
date_train_l = merge_df_l.loc[date_train_l_mask, :]
date_test_l = merge_df_l.loc[date_test_l_mask, :]

date_test_d.drop('일시', axis=1, inplace=True)
date_train_d.drop('일시', axis=1, inplace=True)
date_test_l.drop('일시', axis=1, inplace=True)
date_train_l.drop('일시', axis=1, inplace=True)

date_test_d = date_test_d[["날짜",'요일','기온(°C)','강수량(mm)','풍속(m/s)',"습도(%)",'적설(cm)']]
date_train_d = date_train_d[["날짜",'요일','기온(°C)','강수량(mm)','풍속(m/s)',"습도(%)",'적설(cm)']]
date_test_l = date_test_l[["날짜",'요일','기온(°C)','강수량(mm)','풍속(m/s)',"습도(%)",'적설(cm)']]
date_train_l = date_train_l[["날짜",'요일','기온(°C)','강수량(mm)','풍속(m/s)',"습도(%)",'적설(cm)']]

date_train_d.to_csv('save/data/preprocess/date_train_dinner.csv', index=False)
date_test_d.to_csv('save/data/preprocess/date_test_dinner.csv', index=False)
date_train_l.to_csv('save/data/preprocess/date_train_lunch.csv', index=False)
date_test_l.to_csv('save/data/preprocess/date_test_lunch.csv', index=False)

# merge_df_l.reset_index(drop=False, inplace=True)
# merge_df_d.reset_index(drop=False, inplace=True)

# merge_df_l.drop('index', axis=1, inplace=True)
# merge_df_d.drop('index', axis=1, inplace=True)
