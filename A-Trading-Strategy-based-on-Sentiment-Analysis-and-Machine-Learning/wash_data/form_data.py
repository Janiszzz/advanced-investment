import pandas as pd
import numpy as np
import string
import os
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from snownlp import SnowNLP
import datetime as dt

#%%格式
os.chdir(r'D:\Janis\Desktop\研一上\大数据\final\data')
post = pd.read_pickle('guba1205.pkl')
a = pd.read_pickle('guba6.pkl')
post = post.append(a)
post = post.drop_duplicates()

post['time'] = post['time'].astype(str)
post['time'] = '2022-'+post['time']
post['time'] = pd.to_datetime(post['time'],format='%Y-%m-%d %H:%M', errors='coerce')

post = post[post['time']<dt.datetime(2022,12,1)]

def calc_next(group):
    group.loc[group['time'].dt.hour > 15,'time']+=dt.timedelta(days=1)
    return group    
post = post.groupby('stkcd').apply(calc_next)
post = post.reset_index(drop=True)

post['read'] = post['read'].replace(to_replace='万', value='e+04', regex=True)
post['comment'] = post['comment'].replace(to_replace='万', value='e+04', regex=True)
#%%return格式
raw = pd.read_pickle('hushi.pkl')
ret = raw
def calc_ret(group):
    group['ret'] = np.log((group['Clsprc']/group['Clsprc'].shift(1)).astype(float))
    return group
ret = ret.groupby('Stkcd').apply(calc_ret)
ret = ret[['Stkcd','Trddt','ret','Dnshrtrd','Dsmvosd']]
ret.columns = ['stkcd','date','ret','volume','mktvalue']

ret = ret.reset_index(drop=True)
ret['date'] = pd.to_datetime(ret['date'])
ret = ret.rename(columns={'date':'time'})
#%%
post.to_pickle('post.pkl')
ret.to_pickle('ret.pkl')





