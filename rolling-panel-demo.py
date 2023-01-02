import os
import datetime as dt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression, Lasso, lasso_path, lars_path, LassoLarsIC
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from numpy_ext import rolling_apply
from sklearn.metrics import r2_score
from sklearn.linear_model import LassoCV
import copy

test = [i for i in range(0,122)]
t = [dt.datetime.strftime(x,'%Y-%m-%d') for x in list(pd.date_range(start='20200101', end='20200501'))]
test = pd.DataFrame([test,t])
test = test.T
test.columns = ['ret','date']
test['ret'] = test['ret']/10
for i in range(1,10):
    testa = copy.copy(test)
    testa['id'] = i
    testa['ret'] += i
    test = test.append(testa)
test = test.dropna()

def f2(group):
    print(test.loc[group.index,'ret'].sum())
    return 0
test.sort_values(by=['date','id'])
test = test.reset_index(drop=True)
step = len(test['id'].unique())

test['ret'].rolling(window = 10*step, step = step).apply(f2)