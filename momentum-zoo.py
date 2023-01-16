import pandas as pd
import os
import numpy as np
from scipy import stats
import datetime
#%%
os.chdir('D:\Janis\Desktop\momentum')
import janis
raw = pd.read_pickle('1.pkl')
raw = raw.drop_duplicates()
raw = raw.fillna(0)
raw['ret'] = raw['ret'] + 1
raw['ret'] = np.log(raw['ret'])
raw = raw.set_index(['stkcd','date'])
#%%
ret = raw
def f(group):
    group['bm'] = group['bm'].shift(12)
    group['size'] = group['size'].shift(1)
    return group
ret = ret.groupby('stkcd').apply(f)
#%%
ret_temp = ret['ret']
ret = ret.reset_index()
ret = janis.add_mkt_ret(ret)
#%%
temp = ret_temp.groupby('stkcd').rolling(12).sum()
temp = temp.reset_index(drop=True)
ret['mom'] = temp
def f(group):
    group['mom'] = group['mom'] - group['ret'].shift(1)
    return group

ret['mom'] = ret['mom'] - ret['ret']
ret = ret.groupby('stkcd').apply(f)
#%%
a,b = janis.single_sort(ret.dropna(), 'mom', np.arange(0,1.1,0.1))
a,b = janis.single_sort(ret.dropna(), 'size', np.arange(0,1.1,0.1))
a,b = janis.single_sort(ret.dropna(), 'bm', np.arange(0,1.1,0.1))
#%%
a,b,c = janis.double_sort(ret.dropna(), ['size','bm'], np.arange(0,1.2,0.2), np.arange(0,1.2,0.2))

fct = janis.double_gen_fct(ret.dropna(), ['size','bm'], [0,.5,1],[0,.3,.7,1])
fct['lsp_size'] = -fct['lsp_size']
fct.to_pickle('fct.pkl')

a,b,c = janis.double_sort(ret.dropna(), ['mom','size'], np.arange(0,1.2,0.2), np.arange(0,1.2,0.2))

a,b,c = janis.double_sort(ret.dropna(), ['mom','bm'], np.arange(0,1.2,0.2), np.arange(0,1.2,0.2))
#%%
fct = fct.sort_values('date')
fct = fct.set_index('date')
fct.dropna().cumsum().plot()
#%%TSMOM
ret_temp = ret[['stkcd','date','ret']]
ret_temp = ret_temp.set_index(['stkcd','date'])
def f2(group):
    day_vol = group.ewm(ignore_na=False,
                          adjust=True,
                          com=60,   
                          min_periods=0).std(bias=False)
    group['sigma'] = day_vol * 252 # annualise
    return group
ret_temp = ret_temp.groupby('stkcd').apply(f2)
ret_temp['modif_ret'] = ret_temp['ret']/ret_temp['sigma']
ret_temp = ret_temp[['sigma','modif_ret']]
ret = pd.merge(ret,ret_temp,on = ['stkcd','date'])

import statsmodels.api as sm
#%%
ans = pd.DataFrame()

for i in range(1,49):
    ret['lag_'+str(i)+'_mr'] = ret['modif_ret'].shift(i)
    model = sm.OLS(ret.dropna()['modif_ret'],sm.add_constant(ret.dropna()['lag_'+str(i)+'_mr']))
    ans = ans.append([model.fit().params,   model.fit().tvalues])
#%%
import matplotlib.pyplot as plt
import seaborn as sns

t = ans[ans.index==1].drop('const',axis=1).fillna(0).sum().reset_index(drop=True).to_frame()
t.index += 1
t.columns = ['t.Stat.']
t.plot(kind="bar",figsize=(16,5)).get_figure().savefig('1.png')

m =  ans[ans.index==0].drop('const',axis=1).fillna(0).sum().reset_index(drop=True).to_frame()
m.index += 1
m.columns = ['Mean']
m.plot(kind="bar",figsize=(16,5)).get_figure().savefig('2.png')
#%%
tsmom = np.sign(ret['mom'])*0.4/ret['sigma']*ret['ret']

ret['tsmom'] = tsmom
#%%
def f3(group):
    group['tsmom'] = group['tsmom'].shift(1)
    return group
ret = ret.groupby('stkcd').apply(f3)
#%%
a,b = janis.single_sort(ret.dropna(), 'tsmom', np.arange(0,1.1,0.1))

a,b,c = janis.double_sort(ret.dropna(), ['tsmom','mom'], np.arange(0,1.2,0.2), np.arange(0,1.2,0.2))
#%%Res mom

fama = pd.read_pickle('fct.pkl')
fama.columns = ['date','SMB','HML']
calc = pd.merge(raw.reset_index(),fama,on = 'date',how='inner')
calc = janis.add_mkt_ret(calc)
calc = calc[['stkcd', 'date','ret','mkt_mean','SMB','HML']]
#%%偏移ret，避免look ahead bias
def f2(group):
    group['ret'] = group['ret'].shift(1)
    group['date'] = group['date'].shift(1)
    return group
calc = calc.groupby('stkcd').apply(f2)
calc = calc.dropna()
#%%
import statsmodels.formula.api as smf
model = smf.ols(formula = "ret ~ 1 + mkt_mean +SMB + HML", data = calc).fit()
model.params
model.tvalues
model.resid
#%%
res = model.resid.to_frame()
res.columns = ['res']
calc = pd.concat([calc, res],axis=1)

temp = calc.set_index(['stkcd','date'])['res'].to_frame()
temp = temp.groupby('stkcd', as_index=False).rolling(12).sum().drop('stkcd',axis=1)
temp.columns = ['rsmom']
temp = temp.reset_index()

calc = pd.merge(calc,temp,on=['stkcd','date'])
#%%
def f4(group):
    group['rsmom'] = group['rsmom'] - group['res'].shift(1)
    return group

calc['rsmom'] = calc['rsmom'] - calc['res']
calc = calc.groupby('stkcd').apply(f4)
#%%偏移ret，避免look ahead bias
def f2(group):
    group['ret'] = group['ret'].shift(1)
    group['date'] = group['date'].shift(1)
    return group
calc = calc.groupby('stkcd').apply(f2)
calc = calc.dropna()
#%%
a,b = janis.single_sort(calc.dropna(), 'rsmom', np.arange(0,1.1,0.1))

ret = pd.merge(calc[['stkcd','date','rsmom']],ret,on = ['stkcd','date'])

a,b,c = janis.double_sort(ret.dropna(), ['rsmom','mom'], np.arange(0,1.2,0.2), np.arange(0,1.2,0.2))
#%%

temp = pd.read_excel('TRD_Mnth.xlsx')
temp = temp[2:]
temp.columns = ['stkcd','date','volume','ret']
temp['date'] = pd.to_datetime(temp['date'])
ret = pd.merge(ret,temp[['stkcd','date','volume']],on=['stkcd','date'])
ret['mod_ret'] = ret['volume']*ret['ret']
#%%
temp = ret[['stkcd','date','mod_ret']].set_index(['stkcd','date']).groupby('stkcd',as_index=False).rolling(12).sum()
temp = temp.drop('stkcd',axis=1)
temp.columns = ['volume_mom']
ret = pd.merge(ret,temp,on=['stkcd','date'])
#%%
def f(group):
    group['volume_mom'] = group['volume_mom'] - group['mod_ret'].shift(1)
    return group

ret['volume_mom'] = ret['volume_mom'] - ret['mod_ret']
ret = ret.groupby('stkcd').apply(f)
ret['volume_mom'] = abs(ret['volume_mom'])
ret['volume_mom'] = ret['volume_mom'].astype(np.float64)
#%%
a,b = janis.single_sort(ret[['stkcd','date','ret','mkt_mean','volume_mom']].dropna(), 'volume_mom', np.arange(0,1.1,0.1))

a,b,c = janis.double_sort(ret[['stkcd','date','ret','mkt_mean','volume_mom','mom']].dropna(), ['volume_mom','mom'], np.arange(0,1.2,0.2), np.arange(0,1.2,0.2))

ret[['stkcd','date','ret','mkt_mean','volume_mom','mom']].to_pickle('MSF.pkl')







