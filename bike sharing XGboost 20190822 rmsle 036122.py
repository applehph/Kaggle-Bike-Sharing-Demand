# -*- coding: utf-8 -*-
"""
Problem: https://www.kaggle.com/c/bike-sharing-demand/data
Solution: below the code
Result: public score 0.36122

@author: NorwayPing
"""
# Load in our libraries
import pandas as pd
import numpy as np
import xgboost as xgb

import plotly.offline as py
py.init_notebook_mode(connected=True)
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import make_scorer

from mlxtend.regressor import StackingCVRegressor
from catboost import CatBoostRegressor
from sklearn.model_selection import cross_val_score

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



df['holiday'] = df[['Month', 'Day', 'holiday']].apply(lambda x: (x['holiday'], 1)[x['Month'] == 12 and (x['Day'] in [24, 26, 31])], axis = 1)
df['workingday'] = df[['Month', 'Day', 'workingday']].apply(lambda x: (x['workingday'], 0)[x['Month'] == 12 and x['Day'] in [24,26, 31]], axis = 1)



train=df.loc[df.datetime.isin (train_wIndex)]

test=df.loc[df.datetime.isin (test_wIndex)]


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
#    train, test, y_train, y_test = train_test_split(data, data, test_size = 0.25, random_state = 200)
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


def predict_on_validation_set(model, input_cols):
    #data = get_data()
    data = train
    train_split, test_split = custom_train_test_split(data)

    X_train, y_train_r, y_train_c = prep_data(train_split, input_cols)
    X_test, y_test_r, y_test_c = prep_data(test_split, input_cols)

    model_r = model.fit(X_train, y_train_r)
    y_pred_r = np.exp(model_r.predict(X_test)) - 1

    model_c = model.fit(X_train, y_train_c)
    y_pred_c = np.exp(model_c.predict(X_test)) - 1

    y_pred_comb = np.round(y_pred_r + y_pred_c)
    y_pred_comb[y_pred_comb < 0] = 0

    y_test_comb = np.exp(y_test_r) + np.exp(y_test_c) - 2

    score = get_rmsle(y_pred_comb, y_test_comb)
    return (y_pred_comb, y_test_comb, score)


# predict on test set & transform output back from log scale
def predict_on_test_set(model, x_cols):
    # prepare training set
    #df_train = df[df['_data'] == 'train'].copy()
    df_train = train
    X_train = df_train[x_cols].as_matrix()
    y_train_cas = df_train['casual_log'].as_matrix()
    y_train_reg = df_train['registered_log'].as_matrix()

    # prepare test set
    X_test = test[x_cols].as_matrix()

    casual_model = model.fit(X_train, y_train_cas)
    y_pred_cas = casual_model.predict(X_test)
    y_pred_cas = np.exp(y_pred_cas) - 1
    registered_model = model.fit(X_train, y_train_reg)
    y_pred_reg = registered_model.predict(X_test)
    y_pred_reg = np.exp(y_pred_reg) - 1
    y_pred_comb = y_pred_cas + y_pred_reg
    y_pred_comb[y_pred_comb < 0] = 0
    # add casual & registered predictions together
    return y_pred_comb

def log_rmsle(y_actual,y_pred):
    diff = y_pred -y_actual
    mean_error = np.square(diff).mean()
    return -np.sqrt(mean_error)


params = {'n_estimators': 1000, 'max_depth': 20, 'random_state': 0, 'min_samples_split' : 5, 'n_jobs': -1}
rf_model = RandomForestRegressor(**params)
rf_cols =  [
    'weather','atemp', 'humidity', 'windspeed',
    'holiday', 'workingday', 
    'Hour', 'week_day', 'Year','Day', 'season'
    ]

params = {'n_estimators': 150, 'max_depth': 5, 'random_state': 0, 'min_samples_leaf' : 10, 'learning_rate': 0.1, 'subsample': 0.7, 'loss': 'ls'}
gbm_model = GradientBoostingRegressor(**params)
gbm_cols = [
    'weather','atemp', 'humidity', 'windspeed',
    'holiday', 'workingday', 
    'Hour', 'week_day', 'Year','Day', 'season'
    ]

xgb_model = xgb.XGBRegressor(n_estimators=1000,max_depth= 5, learning_rate = 0.03 ,colsample_bytree =0.8, subsample=0.7,booster = 'gbtree')
xgb_cols = [
    'weather','atemp', 'humidity', 'windspeed',
    'holiday', 'workingday', 
    'Hour', 'week_day', 'Year','Day', 'season'
    ]


params = {'depth': 6, 'learning_rate': 0.05, 'iterations': 150}
cat_model = CatBoostRegressor(1000)
cat_model.fit(train[xgb_cols],train['registered_log'])  


lr = LinearRegression()
streg_model = StackingCVRegressor(regressors=[cat_model,rf_model,gbm_model, xgb_model], meta_regressor=lr)


scores_casual_cat = cross_val_score(cat_model,train[xgb_cols],train['casual_log'], cv=5, scoring=make_scorer(log_rmsle,greater_is_better=False)) 
scores_r_cat = cross_val_score(cat_model,train[xgb_cols],train['registered_log'], cv=5, scoring=make_scorer(log_rmsle,greater_is_better=False)) 


scores_casual_xgb = cross_val_score(xgb_model,train[xgb_cols],train['casual_log'], cv=5, scoring=make_scorer(log_rmsle,greater_is_better=False)) 
scores_r_xgb = cross_val_score(xgb_model,train[xgb_cols],train['registered_log'], cv=5, scoring=make_scorer(log_rmsle,greater_is_better=False)) 

scores_c_rf = cross_val_score(rf_model,train[xgb_cols],train['casual_log'], cv=5, scoring=make_scorer(log_rmsle,greater_is_better=False)) 
scores_r_rf = cross_val_score(rf_model,train[xgb_cols],train['registered_log'], cv=5, scoring=make_scorer(log_rmsle,greater_is_better=False)) 

scores_casual_gbm = cross_val_score(gbm_model,train[xgb_cols],train['casual_log'], cv=5, scoring=make_scorer(log_rmsle,greater_is_better=False)) 
scores_r_gbm = cross_val_score(gbm_model,train[xgb_cols],train['registered_log'], cv=5, scoring=make_scorer(log_rmsle,greater_is_better=False)) 

scores_casual_xgb = cross_val_score(xgb_model,train[xgb_cols],train['casual_log'], cv=5, scoring=make_scorer(log_rmsle,greater_is_better=False)) 
scores_r_xgb = cross_val_score(xgb_model,train[xgb_cols],train['registered_log'], cv=5, scoring=make_scorer(log_rmsle,greater_is_better=False)) 


#scores_casual_streg = cross_val_score(streg_model,train[xgb_cols],train['casual_log'], cv=5, scoring=make_scorer(log_rmsle,greater_is_better=False)) 
#scores_r_streg = cross_val_score(streg_model,train[xgb_cols],train['registered_log'], cv=5, scoring=make_scorer(log_rmsle,greater_is_better=False)) 


(gbm_p, gbm_t, gbm_score) = predict_on_validation_set(gbm_model, gbm_cols)
(xgb_p, xgb_t, xgb_score) = predict_on_validation_set(xgb_model, gbm_cols)
(rf_p, rf_t, rf_score) = predict_on_validation_set(rf_model, gbm_cols)
(streg_p, streg_t, streg_score) = predict_on_validation_set(streg_model, gbm_cols)
(cat_p, cat_t, cat_score) = predict_on_validation_set(cat_model, gbm_cols)

xgb_pred = predict_on_test_set(xgb_model, xgb_cols)
cat_pred = predict_on_test_set(cat_model, xgb_cols)
y_comb_pred = predict_on_test_set(streg_model, xgb_cols)


#ab_pred = predict_on_test_set(ab_model, ab_cols)
#y_pred = np.round(gbm_pred )
#y_pred = np.round(.2*rf_pred + .8*gbm_pred)
#y_pred = np.round(.1*rf_pred + .2*gbm_pred+.7*xgb_pred)
#y_pred = xgb_pred
#y_pred = cat_pred
y_pred = y_comb_pred
#y_pred = np.round(rf_pred )

# output predictions for submission
test['count'] = y_pred
test['datetime'] = df.loc[df.datetime.isin (test_wIndex)]['DateForm']
final_df = test[['datetime', 'count']].copy()
final_df.to_csv('submit_bike_sharing.csv', index=False)