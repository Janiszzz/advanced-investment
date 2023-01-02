import pandas as pd
import numpy as np
import string
import os
import nltk
from snownlp import SnowNLP
import datetime as dt
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import linear_model
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from xgboost import plot_importance
import copy
#%%
os.chdir(r'D:\Janis\Desktop\研一上\大数据\final\data')
#%%
calc = pd.read_pickle('calc1206.pkl')
calc = calc.dropna()
calc['direction'] = np.sign(calc['ret']).astype(int)
calc['volume'] = calc['volume'].astype(int)
calc['mktvalue'] = calc['mktvalue'].astype(float)
calc['random'] = np.random.rand(len(calc['mktvalue']))

calc_train = calc[calc['time']<=dt.datetime(2022,8,31)]
calc_forcast = calc[calc['time']>dt.datetime(2022,8,31)]

#fct_l = ['volume', 'mktvalue', 'lag_ret']
#fct_l = ['volume', 'mktvalue', 'sentiment','lag_ret']
#fct_l = ['volume', 'mktvalue', 'random','lag_ret']
#fct_l = ['lag_ret']
fct_l = ['volume', 'mktvalue', 'read', 'random','comment', 'count_zh', 'count_di','lag_ret']
#fct_l = ['volume', 'mktvalue', 'read', 'comment','lag_ret']
#fct_l = ['volume', 'mktvalue', 'read', 'comment', 'count_zh', 'count_di','lag_ret']
#fct_l = ['volume', 'mktvalue', 'read', 'comment', 'sentiment', 'count_zh', 'count_di','lag_ret']
#fct_l = [ 'read', 'comment', 'sentiment', 'count_zh', 'count_di','lag_ret']
#fct_l = [ 'sentiment','lag_ret']
#%%

def do_model(group, fct_l, _model, name):
    
    X = group[fct_l]
    y = group['direction']
    
    train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.33, random_state=0)
    
    model = _model
    model.fit(train_x, train_y)
    train_pred = model.predict(train_x)
    test_pred = model.predict(test_x)

    group['train_accu_'+name] = accuracy_score(train_y, train_pred)
    group['test_accu_'+name] = accuracy_score(test_y, test_pred)
  
    return group    
#%%
def do_xgb(group, fct_l):
    params = {
        'booster':'gbtree',
        'objective':'multi:softmax',
        'num_class':3,
        'gamma':0.1,
        'max_depth':6,
        'lambda':2,
        'subsample':0.7,
        'colsample_bytree':0.7,
        'min_child_weight':3,
        'slient':1,
        'eta':0.1,
        'seed':1000,
        'nthread':4,
    }
    
    X = group[fct_l]
    y = group['direction']
    
    train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.33, random_state=0)
    
    plst = list(params.items())
    
    dtrain = xgb.DMatrix(train_x, train_y+1)
    num_rounds = 500
    
    model = xgb.train(plst,dtrain,num_rounds)
    test_pred = model.predict(xgb.DMatrix(test_x))
    train_pred = model.predict(xgb.DMatrix(train_x))

    group['train_accu_xgb'] = accuracy_score(train_y, train_pred-1)
    group['test_accu_xgb'] = accuracy_score(test_y, test_pred-1)
    
    return group
#%%
models = [
    ['gauss',GaussianNB()],
    ['forest',RandomForestClassifier(n_estimators=10)],
    ]
#%%
calc_train1 = calc_train
for name, model in models:
    calc_train1 = calc_train1.groupby('stkcd').apply(do_model,fct_l,model,name)

calc_train1 = calc_train1.groupby('stkcd').apply(do_xgb,fct_l)
#%%
dick = calc_train1.groupby('stkcd').mean()
dick = dick[['test_accu_gauss', 'test_accu_forest','test_accu_xgb']]
bestm = dick.idxmax(1)
bestm = bestm.replace(to_replace='test_accu_', value='', regex=True)
#%%

def predict(group, fct_l, _model, name):
    
    X = group[fct_l]
    y = group['direction']
    
    train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.33, random_state=0)
    
    model = _model
    model.fit(train_x, train_y)
    train_pred = model.predict(train_x)
    test_pred = model.predict(test_x)

    group['train_accu_'+name] = accuracy_score(train_y, train_pred)
    group['test_accu_'+name] = accuracy_score(test_y, test_pred)
 
    return group    
#%%
bestm = bestm.reset_index()
bestm.columns=['stkcd','bestm']
calc1 = pd.merge(calc,bestm,on='stkcd')
#%%
def predict(group, fct_l):
    try:
        _model = group['bestm'].to_list()[0]
        if(_model == 'gauss'):
            
            train_x = group[group['time']<=dt.datetime(2022,8,31)]
            train_y = train_x['direction']
            train_x = train_x[fct_l]
            
            forecast_x = group[group['time']>dt.datetime(2022,8,31)]
            timeline = forecast_x['time']
            
            forecast_x = forecast_x[fct_l]
            model = GaussianNB()
            model.fit(train_x, train_y)
        
            position = model.predict(forecast_x)
            position = pd.DataFrame(position,columns=['position'])
            timeline = timeline.reset_index(drop=True)
            position = pd.concat([timeline,position],axis=1, ignore_index=True)
            position.columns = ['time','position']
            group = pd.merge(group,position, on = 'time')
            
        elif(_model == 'forest'):
            
            train_x = group[group['time']<=dt.datetime(2022,8,31)]
            train_y = train_x['direction']
            train_x = train_x[fct_l]
            
            forecast_x = group[group['time']>dt.datetime(2022,8,31)]
            timeline = forecast_x['time']
            
            forecast_x = forecast_x[fct_l]
            model = RandomForestClassifier(n_estimators=10)
            model.fit(train_x, train_y)
        
            position = model.predict(forecast_x)
            
            position = pd.DataFrame(position,columns=['position'])
            timeline = timeline.reset_index(drop=True)
            position = pd.concat([timeline,position],axis=1, ignore_index=True)
            position.columns = ['time','position']
            group = pd.merge(group,position, on = 'time')
            
        elif(_model == 'xgb'):
            params = {
                'booster':'gbtree',
                'objective':'multi:softmax',
                'num_class':3,
                'gamma':0.1,
                'max_depth':6,
                'lambda':2,
                'subsample':0.7,
                'colsample_bytree':0.7,
                'min_child_weight':3,
                'eta':0.1,
                'seed':1000,
                'nthread':4,
            }
            
            
            train_x = group[group['time']<=dt.datetime(2022,8,31)]
            train_y = train_x['direction']
            train_x = train_x[fct_l]
            
            forecast_x = group[group['time']>dt.datetime(2022,8,31)]
            timeline = forecast_x['time']
            
            forecast_x = forecast_x[fct_l]
           
            plst = list(params.items())
            
            dtrain = xgb.DMatrix(train_x, train_y+1)
            num_rounds = 500
            
            model = xgb.train(plst,dtrain,num_rounds)
            position = model.predict(xgb.DMatrix(forecast_x))
            
            position = pd.DataFrame(position,columns=['position'])
            position -= 1
            timeline = timeline.reset_index(drop=True)
            position = pd.concat([timeline,position],axis=1, ignore_index=True)
            position.columns = ['time','position']
            group = pd.merge(group,position, on = 'time')
        
    except:
        group['position'] = 0
    return group   
#%%
calc1 = calc1.groupby('stkcd').apply(predict,fct_l)

R = calc1[calc1['time']>dt.datetime(2022,8,31)]

R = R[['time','ret','position']]

R['R'] = R['ret']*R['position']
mkt = R.groupby('time')['ret'].mean()


RR = R.groupby('time')['R'].mean()
RR = pd.merge(RR,mkt,on='time')
RR.columns=['strategy','market index']
RR.plot(figsize=(16,10))
RR.cumsum().plot(figsize=(16,10))
RR.describe()
RR.cumsum().describe()


