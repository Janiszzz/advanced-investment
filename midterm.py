import numpy as np
import pandas as pd
import os
import datetime
from numpy_ext import rolling_apply
#%%
os.chdir('D:\Janis\Desktop\研一上\资产定价（投资学）\midterm')
import janis
#%%
pkl = pd.read_pickle('ASharePrices20200309.pkl')
ret = pkl[['S_INFO_WINDCODE', 'TRADE_DT','S_DQ_ADJCLOSE']]
ret.columns = ['windcode','date','clsprc']
ret['date'] = pd.to_datetime(ret['date'],format='%Y%m%d')
#%%
def f(group):
    group['ret'] = np.log(group['clsprc']/group['clsprc'].shift(1))
    return group
ret = ret.groupby('windcode').apply(lambda x: x.sort_values('date', ascending=True))
ret = ret.reset_index(drop=True)
#%%
ret = ret.groupby('windcode').apply(f)
ret = ret.dropna()
ret = janis.add_mktret(ret)
ret = janis.gb_vol(ret, 'windcode', 20)
ret = janis.downside_beta(ret, 'windcode', 20)
ret = ret.reset_index(drop=True)

#%%

janis.single_sort(ret, 'G', 5)
janis.single_sort(ret, 'B', 5)



ret = ret.dropna()

ret = janis.double_sort(ret, ['G','db'], 10, 10)
janis.print_sort(ret, ['G','db'])
janis.double_sort(ret, ['B','db'], 10, 10)
janis.print_sort(ret, ['B','db'])

#%%
def f(group): 
    group['cumret'] = group['ret'].rolling(5).sum()
    return group
ret = ret.groupby('windcode').apply(f)
#%%
def f2(group): 
    group['cumvol'] = group['ret'].rolling(20).std()
    return group
ret = ret.groupby('windcode').apply(f2)
#%%
ret['ret'] = ret['ret'].shift(-1)
ret = ret.dropna()
a,b,c = janis.fama_macbeth(ret, 'windcode', 'l', ['G','B','cumret','cumvol'], 'ret')
#%%
fct1 = janis.gen_fct(ret, ['G','db'], [0,.2,.4,.6,.8,1],[0,.2,.4,.6,.8,1])
fct2 = janis.gen_fct(ret, ['B','db'], [0,.2,.4,.6,.8,1],[0,.2,.4,.6,.8,1])

fama = pd.merge(ret,fct1,on ='date')

janis.neu(fama,['ret','G_factor'])
janis.neu(fama,['resid_ret_G_factor','db_factor'])

fama = pd.merge(ret,fct2,on ='date')

janis.neu(fama,['ret','B_factor'])
janis.neu(fama,['resid_ret_B_factor','db_factor'])










#%%
'''
中性化：
beta- ~ 1+SUE+residual
residual称为中性化后的

因子等价：
ret ~ 1+SUE+residual
residual ~ 1+beta-
如果系数不显著则说明两个解释力相当，等价

'''






