import plotly.graph_objects as go
import numpy as np
import os
import datetime as dt
import pandas as pd
import matplotlib.pyplot as plt
import mplfinance as mpf
from mplfinance.original_flavor import candlestick_ohlc
import matplotlib.gridspec as gridspec
#%%
os.chdir(r"D:\Downloads\OHLC")

df = pd.read_pickle("ohlcdata.pkl")
df = df.rename(columns={'index':'date'})
df['date'] = pd.to_datetime(df['date'])
df[['openPrice', 'highestPrice', 'lowestPrice','closePrice']] = np.float32(df[['openPrice', 'highestPrice', 'lowestPrice','closePrice']])

import janis
df = janis.calc_ret(df,'cont','closePrice')
df = df.sort_values('date', ascending=True)

import torch
pos = torch.load("ans.pt")
pos = pd.DataFrame(pos)
pos.columns = ['cont','date','real','step','up','down']
pos = pos[['cont','date','up']]
pos['date'] = pd.to_datetime(pos['date'])
pos = pos.sort_values('date', ascending=True)
#%%
def f3(group):
    group['ret'] = group['ret'].rolling(60).sum()
    group['ret'] = group['ret'].shift(-60)
    return group
df = df.groupby('cont').apply(f3)
#df = df.loc[df['date'] > dt.datetime(2021,1,2)]
df = df.rename(columns = {'ret':'future60ret'})
df.groupby('date').mean()['future60ret'].plot(figsize = (16,10))
#%%
def f(group):
    group['date'] = group['date'].shift(-60)
    return group
pos = pos.groupby('cont').apply(f)
#%%
calc = pd.merge(pos,df,on = ['date','cont'] ,how = 'inner')
#%%
calc = calc.rename(columns = {'future60ret':'ret'})
calc = janis.add_mkt_ret(calc)
a,c = janis.single_sort(calc,'up', np.arange(0,1.1,0.1))
c = pd.DataFrame(c)

pos = pd.merge(pos,df,on = ['date','cont'] ,how = 'inner')
#%%
def f2(group):
    group['fct_ret'] = group.loc[group['up_type']==4,'ret'].mean() - group.loc[group['up_type']==0,'ret'].mean()
    return group
fct_ret = a.groupby('date').apply(f2)[['date','fct_ret']].drop_duplicates()
fct_ret = fct_ret.set_index('date')/60
#%%

mkt = df[['date','mkt_mean']].drop_duplicates()

figure = pd.merge(mkt,fct_ret,on = 'date').dropna()
figure.plot(figsize = (16,10))
figure.columns = ['market index', 'long-short ptf return']
(figure.cumsum()*100).plot(xlabel = 'date', ylabel = 'return rate(%)', figsize = (16,10))

figure.mean()/figure.std()












