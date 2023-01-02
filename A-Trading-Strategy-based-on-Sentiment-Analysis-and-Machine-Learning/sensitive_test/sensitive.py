import pandas as pd
import numpy as np
import string
import os
import nltk
from snownlp import SnowNLP
import datetime as dt
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import linear_model
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from xgboost import plot_importance
import copy

#%%
os.chdir(r'D:\Janis\Desktop\研一上\大数据\final\data')
import janis
#%%
calc = pd.read_pickle('calc1206.pkl')
calc = calc.dropna()
calc['direction'] = np.sign(calc['ret']).astype(int)
calc['volume'] = calc['volume'].astype(int)
calc['mktvalue'] = calc['mktvalue'].astype(float)
calc['random'] = np.random.rand(len(calc['mktvalue']))
#%%
def m(group):
    group['mktret'] = group['ret'].mean()
    return group
calc = calc.groupby('time').apply(m)


#%%
calc = calc[['time','ret','mktvalue','sentiment','mktret']]
calc.columns = ['date','ret','mktvalue','sentiment','mktret']
df = janis.gen_fct(calc, ['mktvalue','sentiment'], [0,0.5,1], [0,0.5,1])
janis.print_sort(calc, ['mktvalue','sentiment'])
#%%
calc = pd.merge(calc,df,on='date')

a,b,c = janis.fama_macbeth(calc, 'stkcd', 1, ['mktret','mktvalue_factor', 'sentiment_factor'], 'ret')


