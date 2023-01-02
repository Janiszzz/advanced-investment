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
#%%
os.chdir(r'D:\Janis\Desktop\研一上\资产定价（投资学）\final')
import janis
#%%
raw = pd.read_pickle('ASharePrices20200309.pkl')
#%%
ret = raw
ret = ret.drop_duplicates()
ret.columns = ['windcode','date','volume','amount','adjopen','adjhigh','adjlow','adjclose','open']
ret['date'] = pd.to_datetime(ret['date'],format='%Y%m%d')

ret = janis.calc_ret(ret, 'windcode', 'adjclose')
ret = ret[['windcode','date','ret']]


def f(group):
    mm = MinMaxScaler()
    ret = np.array(group['ret']).reshape(-1,1)
    group['ret'] = mm.fit_transform(ret).reshape(-1)
    return group
ret = ret.groupby('date').apply(f)

#ret.loc[ret['date']==dt.datetime(2020,1,2),'ret'].describe()
ret = ret.dropna()
ret = ret[ret['windcode']<='002032.SZ']

from time import time
import warnings
warnings.filterwarnings('ignore')
retp = ret.pivot(index='date', columns='windcode', values='ret')
retp = retp.fillna(0)

window = 20

predvalues = np.zeros((retp.shape[0], retp.shape[1]))
for i in range(window,retp.shape[0]):
    print(i)
    X = retp.iloc[i-window:i,:]
    start = time()
    if(X.shape[0]<window):
        continue
    for j in range(retp.shape[1]):
        x = np.hstack([X.iloc[:, :j],X.iloc[:, j+1:]])
        y = X.iloc[:, j]
        alpha = LassoCV(n_jobs=-1).fit(x, y).alpha_
        coef = Lasso(alpha = alpha).fit(x,y).coef_
        predcols = [i for i in range(x.shape[1]) if coef[i] !=0]
        if np.sum(coef) == 0:
            predvalues[i,j] = 0
        else:
            predvalues[i,j] = LinearRegression(n_jobs=-1).fit(x[:, predcols], y).predict(x[-1, predcols].reshape(1, -1))
    end = time()    
    print('%.3f seconds' % (end - start))        

predvalues.name = 'SO'
predvalues = pd.DataFrame(predvalues)
predvalues = predvalues.reset_index(drop=False)

predvalues.to_pickle('predvalues.pkl')



