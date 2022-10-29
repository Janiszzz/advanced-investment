import pandas as pd
import numpy as np
import os
from scipy import stats

#%%
os.chdir('D:\Janis\Desktop\研一上\资产定价（投资学）\A7')
msf = pd.read_sas('MSF.sas7bdat')
shrcd = pd.read_sas('permno_shrcd.sas7bdat')
raw_comp = pd.read_sas('compann.sas7bdat')
g2p = pd.read_sas('crspcusip.sas7bdat')
#%%
ret = pd.merge(msf, shrcd, on = 'PERMNO')
ret = ret[(ret['SHRCD']>=10)&(ret['SHRCD']<= 12)&(ret['HEXCD']==1)]
#计算当月market equity
ret['mkt_v'] = abs(ret['PRC'])*ret['SHROUT']
ret = ret[['PERMNO','DATE','RET','mkt_v']]
ret.rename(columns={'PERMNO':'Permno','DATE':'date','RET':'ret'},inplace=True)
g2p = g2p[['Permno','Gvkey']]
ret = pd.merge(ret, g2p, on = 'Permno')
ret['year'] = ret['date'].dt.year
#%%
#https://www.jianshu.com/p/eba1aaad5fa0
#https://zhuanlan.zhihu.com/p/149067136
comp = raw_comp
comp = comp[(comp['INDFMT']==b'INDL')&(comp['DATAFMT']==b'STD')&(comp['POPSRC']==b'D')&(comp['CONSOL']==b'C')]
comp['GVKEY'] = comp['GVKEY'].apply(int)
comp.rename(columns={'GVKEY':'Gvkey','DATADATE':'date'},inplace=True)
#计算12月book equity
comp['book']=comp['SEQ']+comp['TXDB']-comp['PSTK']
comp = comp.dropna()
comp['month'] = comp['date'].dt.month
comp = comp[comp['month'] == 12]
comp['year'] = comp['date'].dt.year
comp=comp[['Gvkey','year','book']]
#%%
#合并以上，取交集
#按year合并，就会使得当年所有日期下的book都是12月的
calc = pd.merge(ret,comp,how = 'inner', on=['Gvkey','year'])
calc.drop_duplicates(inplace=True)
calc = calc.dropna(subset = 'Permno')
#计算book to market 
#book是12月的，而mkt value却是当月的
calc['btm'] = calc['book']/calc['mkt_v']
#只取btm>0的股票
calc = calc[calc['btm']>0]
calc = calc[['Permno','date','ret','mkt_v','btm','year']]
calc['month'] = calc['date'].dt.month

#%%标注HMLBS
#https://zhuanlan.zhihu.com/p/55071842
def f1(group):
    group['type_size'] = group['mkt_v'].map(lambda x: 'B' if x >= group['mkt_v'].median() else 'S')
    
    border_down, border_up = group['btm'].quantile([0.3, 0.7])
    group['type_btm'] = group['btm'].map(lambda x: 'H' if x >= border_up else 'M')
    group['type_btm'] = group.apply(lambda row: 'L' if row['btm'] <= border_down else row['type_btm'], axis=1)
    return group

calc = calc.groupby('date').apply(f1)

#%%计算每个月的minus
def f(group):
    SH=group.query('(type_size=="S")&(type_btm=="H")')
    SM=group.query('(type_size=="S")&(type_btm=="M")')
    SL=group.query('(type_size=="S")&(type_btm=="L")')
    BH=group.query('(type_size=="B")&(type_btm=="H")')
    BM=group.query('(type_size=="B")&(type_btm=="M")')
    BL=group.query('(type_size=="B")&(type_btm=="L")')
    
    R_SH=SH['ret'].mean()
    R_SM=SM['ret'].mean()
    R_SL=SL['ret'].mean()
    R_BH=BH['ret'].mean()
    R_BM=BM['ret'].mean()
    R_BL=BL['ret'].mean()
    
    group['SMB_m'] = (R_SL + R_SM + R_SH - R_BL - R_BM - R_BH) / 3
    group['HML_m'] = (R_SH + R_BH - R_SL - R_BL) / 2
        
    return group
calc = calc.groupby('date').apply(f)
calc = calc.drop_duplicates('date')
calc = calc.dropna()
#截至，12个月的和
calc['SMB'] = calc['SMB_m'].rolling(12).mean()
calc['HML'] = calc['HML_m'].rolling(12).mean()

#%%
compare = calc[['date','SMB_m','HML_m']]
compare = compare.drop_duplicates('date')
compare['date'] = compare['date'].dt.year*100+compare['date'].dt.month

ff = pd.read_csv('F-F_Research_Data_Factors.CSV')
ff = ff.dropna()
ff = ff.astype(float)
ff['date'] = ff['date'].astype(int)
compare = pd.merge(compare,ff,on = 'date')
compare = compare[['SMB_m','SMB','HML_m','HML']]
compare.corr()


































