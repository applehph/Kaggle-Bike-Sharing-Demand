# -*- coding: utf-8 -*-
"""
Problem: https://www.kaggle.com/c/bike-sharing-demand/data
Solution: below the code
Result: public score 0.35599

@author: NorwayPing
"""
# Load in our libraries
import pandas as pd
import numpy as np

import plotly.offline as py
py.init_notebook_mode(connected=True)
from sklearn.model_selection import cross_val_predict

from catboost import CatBoostRegressor
import matplotlib.pyplot as plt
import seaborn as sns

# Load in the train and test datasets
train = pd.read_csv('train.csv',delimiter=",", encoding='utf-8')
test = pd.read_csv('test.csv')

train_wIndex = train['datetime']
test_wIndex = test['datetime']


df = train.append(test)

for col in ['casual', 'registered', 'count']:
    df['%s_log' % col] = np.log(df[col] + 1)


# Some features of my own that I have added in
# split the date and the hour
df['DateForm']= pd.to_datetime(df['datetime'])
df['Time'],df['Date']= df['DateForm'].apply(lambda x:x.time()), df['DateForm'].apply(lambda x:x.date())

df['week_day'] = df['Date'].map(lambda x: x.weekday() + 1)

df['Year'] = df['Date'].map(lambda x: x.year)
df['Month'] = df['Date'].map(lambda x: x.month)
#df['Year_Month'] = df['Date'].map(lambda x: 100*x.year + x.month)
df['Day'] = df['Date'].map(lambda x: x.day)
df['Hour'] = df['Time'].map(lambda x: x.hour)


# BlackFriday
df.loc[(df['Year'] == 2011) & (df['Month']== 11) &( df['Day'] == 25 ), "workingday"] = 0
df.loc[(df['Year'] == 2012) & (df['Month']== 11) &( df['Day'] == 23 ), "workingday"] = 0
df.loc[(df['Year'] == 2011) & (df['Month']== 11) &( df['Day'] == 25 ), "holiday"] = 1
df.loc[(df['Year'] == 2012) & (df['Month']== 11) &( df['Day'] == 23 ), "holiday"] = 1
df.loc[(df['Year'] == 2012) & (df['Month']== 12) &( df['Day'] == 24 ), "workingday"] = 0
df.loc[(df['Year'] == 2012) & (df['Month']== 12) &( df['Day'] == 24 ), "holiday"] = 1


#sandy
df['holiday'] = df[['Month', 'Day', 'holiday', 'Year']].apply(lambda x: (x['holiday'], 1)[x['Year'] == 2012 and x['Month'] == 10 and (x['Day'] in [30])], axis = 1)
df['holiday'] = df[['Month', 'Day', 'holiday']].apply(lambda x: (x['holiday'], 1)[x['Month'] == 12 and (x['Day'] in [24, 26, 31])], axis = 1)
df['workingday'] = df[['Month', 'Day', 'workingday']].apply(lambda x: (x['workingday'], 0)[x['Month'] == 12 and x['Day'] in [24,26, 31]], axis = 1)

#There is error on the humidity data on 2011 march 10 which is all 0 in the 24 hours. 
#set it to the mean of the close two days
Mean_humidity = (df.loc[(df['Year'] == 2011) & (df['Month']== 3) &( df['Day'] == 9 )]['humidity'].mean()+df.loc[(df['Year'] == 2011) & (df['Month']== 3) &( df['Day'] == 11 )]['humidity'].mean())/2
df.loc[(df['Year'] == 2011) & (df['Month']== 3) &( df['Day'] == 10 ), "humidity"] = Mean_humidity

#calcualte seson mean casual_log and mean registered_log
df['year_season'] = df[['Year', 'season']].apply(lambda x: (x['season'], 4+x['season'])[x['Year'] == 2012], axis = 1)
by_season_count = df.loc[df.datetime.isin (train_wIndex)].groupby('year_season')[['casual_log']].count()
by_season_count.columns = ['year_season_number']
df = df.join(by_season_count, on='year_season')

by_season = df.loc[df.datetime.isin (train_wIndex)].groupby('year_season')[['casual_log']].agg(sum)
by_season.columns = ['casual_year_season']
df = df.join(by_season, on='year_season')

by_season_r = df.loc[df.datetime.isin (train_wIndex)].groupby('year_season')[['registered_log']].agg(sum)
by_season_r.columns = ['registered_year_season']
df = df.join(by_season_r, on='year_season')

df['averge_c_log'] =df['casual_year_season']/df['year_season_number']
df['averge_r_log'] =df['registered_year_season']/df['year_season_number']


train=df.loc[df.datetime.isin (train_wIndex)][:]

test=df.loc[df.datetime.isin (test_wIndex)][:]

def get_rmsle(y_pred, y_actual):
    diff = np.log(y_pred + 1) - np.log(y_actual + 1)
    mean_error = np.square(diff).mean()
    return np.sqrt(mean_error)


def get_data():
    data = df[df['_data'] == 'train'].copy()
    data = train
    return data


def custom_train_test_split(data, cutoff_day=15):
    train = data[data['Day'] <= cutoff_day]
    test = data[data['Day'] > cutoff_day]
#    train, test, y_train, y_test = train_split(data, data, test_size = 0.25, random_state = 200)
    return train, test


def prep_data(data, input_cols):
    X = data[input_cols].as_matrix()
    y_r = data['registered_log'].as_matrix()
    y_c = data['casual_log'].as_matrix()

    return X, y_r, y_c

def prep_data_casual(data, input_cols,input_cols_casul):
    X = data[input_cols].as_matrix()
    X_casual = data[input_cols_casul].as_matrix()
    y_r = data['registered_log'].as_matrix()
    y_c = data['casual_log'].as_matrix()

    return X,X_casual, y_r, y_c


def predict_on_validation_set_CR(model_c, model_r,input_cols):
    #data = get_data()
    data = train
    train_split, test_split = custom_train_test_split(data)

    X_train, y_train_r, y_train_c = prep_data(train_split, input_cols)
    X_test, y_test_r, y_test_c = prep_data(test_split, input_cols)

    model_r_pred = model_r.fit(X_train, y_train_r)
    y_pred_r = np.exp(model_r_pred.predict(X_test)) - 1

    model_c_pred = model_c.fit(X_train, y_train_c)
    y_pred_c = np.exp(model_c_pred.predict(X_test)) - 1

    y_pred_comb = np.round(y_pred_r + y_pred_c)
    y_pred_comb[y_pred_comb < 0] = 0

    y_test_comb = np.exp(y_test_r) + np.exp(y_test_c) - 2

    score = get_rmsle(y_pred_comb, y_test_comb)
    return (y_pred_comb, y_test_comb, score)

# predict on test set & transform output back from log scale
def predict_on_test_set_CR(model, c_cols,r_cols):
    # prepare training set
    #df_train = df[df['_data'] == 'train'].copy()
    df_train = train
    X_train_c = df_train[c_cols].as_matrix()
    X_train_r = df_train[r_cols].as_matrix()
    y_train_cas = df_train['casual_log'].as_matrix()
    y_train_reg = df_train['registered_log'].as_matrix()

    # prepare test set
    X_test_c = test[c_cols].as_matrix()
    # prepare test set
    X_test_r = test[r_cols].as_matrix()

    casual_model = model.fit(X_train_c, y_train_cas)
    y_pred_cas = casual_model.predict(X_test_c)
    y_pred_cas = np.exp(y_pred_cas) - 1

    registered_model = model.fit(X_train_r, y_train_reg)
    y_pred_reg = registered_model.predict(X_test_r)
    y_pred_reg = np.exp(y_pred_reg) - 1
      
    y_pred_comb = y_pred_cas + y_pred_reg
    y_pred_comb[y_pred_comb < 0] = 0
    # add casual & registered predictions together
    return y_pred_comb

def log_rmsle(y_actual,y_pred):
    diff = y_pred -y_actual
    mean_error = np.square(diff).mean()
    return -np.sqrt(mean_error)


cat_cols = [
    'weather','atemp', 'humidity', 'windspeed',
    'holiday', 'workingday', 
    'Hour', 'week_day', 'Year','Day', 'season', 'averge_r_log',
    ]
cat_cols_casual = [
    'weather','atemp', 'humidity', 'windspeed',
    'holiday', 'workingday', 
    'Hour', 'week_day', 'Year','Day', 'season','averge_c_log',
    ]
params = {'depth': 6, 'learning_rate': 0.05, 'iterations': 150}
cat_model = CatBoostRegressor(1000)



cat_pred_new = predict_on_test_set_CR(cat_model,cat_cols_casual, cat_cols)

y_pred = cat_pred_new
# output predictions for submission
test['count'] = y_pred
test['datetime'] = df.loc[df.datetime.isin (test_wIndex)]['DateForm']
final_df = test[['datetime', 'count']].copy()
final_df.to_csv('submit_bike_sharing.csv', index=False)
