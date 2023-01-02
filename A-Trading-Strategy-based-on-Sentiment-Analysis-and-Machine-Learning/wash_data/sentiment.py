!pip install pandas==1.4.1
import pandas as pd
import numpy as np
import string
import os
import nltk
!pip install snownlp
from snownlp import SnowNLP
import datetime as dt

os.chdir('/content/drive/MyDrive/final')
post = pd.read_pickle('post1206.pkl')
ret = pd.read_pickle('ret.pkl')

def str_count(str):
    count_zh = 0
    count_di = 0
    for s in str:
        if (s in string.ascii_letters)|s.isdigit():
            count_di += 1
        # 中文
        elif s.isalpha():
            count_zh += 1
    return count_zh,count_di
def calc_senti(s):
    senti = SnowNLP(s)
    return senti.sentiments

post['count'] = post['title'].apply(str_count)
post['sentiment'] = post['title'].apply(calc_senti)
df = post['count'].apply(lambda x: pd.Series(x))
df.columns = ['count_zh','count_di']
post['count_zh'] = df['count_zh']
post['count_di'] = df['count_di']

post.to_pickle('calc_post1206.pkl')

a = pd.read_pickle('calc_post.pkl')
post.append(a)

post.to_pickle('calc_post1206_V2.pkl')

post_daily = post[['stkcd','time','read', 'comment', 'sentiment', 'count_zh', 'count_di']]
post_daily.set_index(['time'],inplace=True)
post_daily = post_daily.astype(float)
post_daily = post_daily.groupby('stkcd').resample('1D').mean()
post_daily = post_daily.dropna()

post_daily.drop('stkcd',axis=1)

calc = pd.merge(ret,post_daily.drop('stkcd',axis=1),on = ['stkcd','time'])

def lag_ret(group):
    group['lag_ret'] = group['ret'].shift(1)
    return group
calc = calc.groupby('stkcd').apply(lag_ret)

calc.dropna()

calc.to_pickle('calc1206.pkl')