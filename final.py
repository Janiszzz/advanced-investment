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
so = pd.read_pickle('predvalues.pkl')
#%% calc return and downside beta etc.

ret = raw
ret = ret.drop_duplicates()
ret.columns = ['windcode','date','volume','amount','adjopen','adjhigh','adjlow','adjclose','open']
ret['date'] = pd.to_datetime(ret['date'],format='%Y%m%d')

ret = janis.calc_ret(ret, 'windcode', 'adjclose')
ret = ret[['windcode','date','ret']]
#%%
ret = janis.downside_beta(ret, 'windcode', 20)
#%%
def f(group):
    group['mktret'] = group['ret'].mean()
    return group
ret = ret.groupby('date').apply(f)

#%%add control variables
def vol(df, idl, window):
    def f1(group, window):       
        if(group.shape[0]<window):
            return None
        def f2(ret):
            vol = ret.std()
            return vol
        
        group['sigma'] = rolling_apply(f2, window, group['ret'].values)  
       
        return group

    df = df.groupby(idl).apply(f1,window)
    return df.reset_index(drop=True)
ret = vol(ret,'windcode',20)

cret = ret.groupby('windcode')['ret'].rolling(5).sum()
ret['cret'] = cret.reset_index(drop=True)
ret = janis.add_mkt_ret(ret)
ret = janis.gb_vol(ret, 'windcode', 20)

#%%
calc = pd.merge(ret,so,on = ['windcode','date'], how = 'inner').drop_duplicates()
#%%偏移ret，避免look ahead bias
def f2(group):
    group['ret'] = group['ret'].shift(-1)
    group['date'] = group['date'].shift(-1)
    return group
calc = calc.groupby('windcode').apply(f2)
calc = calc.dropna()
#%%
#1
so.describe()
#2
a,b = janis.single_sort(calc, 'SO', np.arange(0, 1.1, 0.1))
#3
a,b,c= janis.double_sort(calc.dropna(), ['SO','db'], np.arange(0, 1.2, 0.2), np.arange(0, 1.2, 0.2))
#%%
#4
calc = calc.set_index(['windcode', 'date']) 
def fama_macbeth(df, fctl, y):
    def f(fctl):
        fct_string = ''
        for i in fctl:
            fct_string = fct_string + '+'  + i 
        return fct_string
    from linearmodels import FamaMacBeth
    mod = FamaMacBeth.from_formula(y+' ~ 1'+ f(fctl), data=calc)
    res = mod.fit(cov_type= 'kernel',debiased = False, bandwidth = 4)
    print(res.summary)
    return
#%%
#4
fama_macbeth(calc.dropna(),['SO','sigma','cret'], 'ret')
fama_macbeth(calc.dropna(),['sigma_good', 'sigma','cret'], 'ret')
fama_macbeth(calc.dropna(),['sigma_bad','sigma','cret'], 'ret')
fama_macbeth(calc.dropna(),['SO','sigma_good','sigma','cret'], 'ret')
fama_macbeth(calc.dropna(),['SO', 'sigma_bad','sigma','cret'], 'ret')
fama_macbeth(calc.dropna(),['sigma_good', 'sigma_bad','sigma','cret'], 'ret')
fama_macbeth(calc.dropna(),['SO','sigma_good', 'sigma_bad','sigma','cret'], 'ret')
#%%
#5
calc = pd.merge(ret,so,on = ['windcode','date'], how = 'inner').drop_duplicates()
def f2(group):
    group['ret'] = group['ret'].shift(-1)
    group['date'] = group['date'].shift(-1)
    return group
calc = calc.groupby('windcode').apply(f2)
calc = calc.dropna()

fct = janis.gen_fct(calc.dropna(), ['SO','db'], [0,.3,.7,1], [0,.3,.7,1])
fct = janis.neu(fct,['lsp_db', 'lsp_SO'])
calc = pd.merge(calc,fct,on='date',how='inner')
#%%
a,b,c = janis.fama_macbeth(calc.dropna(),['mkt_mean','lsp_SO',  'lsp_db'], 'ret')
a,b,c = janis.fama_macbeth(calc.dropna(),['mkt_mean','lsp_SO'], 'ret')
a,b,c = janis.fama_macbeth(calc.dropna(),['mkt_mean','lsp_db'], 'ret')
a,b,c = janis.fama_macbeth(calc.dropna(),['mkt_mean','resid_lsp_db_lsp_SO'], 'ret')



