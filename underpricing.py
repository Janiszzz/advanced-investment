import pandas as pd
import numpy as np
import os
from scipy import stats
import datetime
import statsmodels.formula.api as smf
#%%
os.chdir('D:\Janis\Desktop\研一上\公司财务\Homework')
#Loughran, T., & Ritter, J. (2004)
ipo = pd.read_excel('hkipo4.xls')
raw = pd.read_excel('hkipo5.xls')

raw = raw.T
raw.columns = raw.iloc[0, :]
raw = raw.iloc[1: , :]
raw = raw.replace(-888,np.nan)
raw = raw.reset_index()
raw['index'] = raw['index'] + 190000
raw = raw.rename(columns={'index':'date'})
raw = raw.set_index('date')
ipo['date'] = ipo['date'] + 19000000
#%%
res = pd.DataFrame()
res['Issue proceeds'] = ipo['offer price']*ipo['Public -  No. of shares ']/1000000
res['Market capitalization'] = ipo[' first day unadjusted price']*ipo['Outstanding shares']/1000000
res['Initial returns'] = ipo[' first day unadjusted price']/ipo['offer price']-1
res['Money left on the table'] = ipo['Public -  No. of shares ']*(ipo[' first day unadjusted price']-ipo['offer price'])/1000000
#%%
def f(s):
    print(s.name,len(s),'%.2f'%s.mean(), '%.2f'%s.median(), '%.2f'%s.max(), '%.2f'%s.min(), '%.2f'%s.std())
res.apply(f)
#%%
ret = raw/raw.shift(1)
ret = ret.replace(np.nan,0)
ret = ret.apply(np.log)

#%%
def f2(s):
    s = s-ret['Heng Seng Index']
    return s
adjret = ret.replace(-np.inf,np.nan).apply(f2)
adjret = adjret.iloc[: , :-2]
ret = ret.iloc[: , :-2]
#%%
figure = pd.DataFrame(columns=['month','IPO cumulative raw returns','Hang Seng Index cumulative raw returns','Hang Seng adjusted cumulative excess returns'])
figure['month'] = range(60)
figure = figure.fillna(0)
figure['month'] = range(60)
#%%
def calc(df):
    temp = pd.DataFrame()
    df = df.replace(-np.inf,np.nan)
    for index, item in df.iteritems():
        temp = temp.append(item.dropna().reset_index(drop=True))
    return temp

figure['IPO cumulative raw returns'] = calc(ret).mean().cumsum()
figure['Hang Seng adjusted cumulative excess returns'] = calc(adjret).mean().cumsum()
figure['Hang Seng Index cumulative raw returns'] = figure['IPO cumulative raw returns'] -figure['Hang Seng adjusted cumulative excess returns']

figure.plot(x='month')

print(figure[0:12].sum(), figure[12:24].sum(), figure[24:36].sum(), figure[36:48].sum(), figure[48:60].sum())





figure.to_excel('figure.xlsx')







