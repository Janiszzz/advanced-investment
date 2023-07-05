import pandas as pd
import datetime as dt
import os
import matplotlib
import numpy as np

#%%
def main_con_adjust(df, key = 'closePrice', price = 'closePrice', method = 'forward_prop'):
  last_main = None
  factor = 1 
  result = pd.Series(dtype='float64')
  flag = pd.Series(dtype='float64')
  
  for date in df['tradeDate'].unique():
    now = df[df['tradeDate']==date]
    main_mark = now.loc[now['mainCon']==1,'ticker'].values[0]
    main_price = now.loc[now['mainCon']==1, key].values[0]
    adjust_price = now.loc[now['mainCon']==1, price].values[0]
    try:
      smain_mark = now.loc[now['smainCon']==1,'ticker'].values[0]
      smain_price = now.loc[now['smainCon']==1, key].values[0]
    except:
      smain_mark = main_mark
      smain_price = main_price
      
    if(not last_main):
      last_main = main_mark
    elif(last_main == smain_mark):
      factor *= smain_price/main_price
      last_main = main_mark
      flag[date] = 1
      
    result[date] = adjust_price*factor
  
  return result,flag

#%%
os.chdir(r"D:\Downloads\OHLC")
raw = pd.read_pickle("future_data_all.pkl")
ret = pd.DataFrame()
#%%
for price in ['openPrice', 'highestPrice', 'lowestPrice', 'closePrice']:
  adj_price_all = pd.DataFrame()
  pos_trans = pd.DataFrame()
  
  for contract in raw['contractObject'].unique():
    df = raw[raw['contractObject']==contract]
    df = df.sort_values(by='tradeDate')
    adj_price, flag = main_con_adjust(df=df,price=price)
    adj_price_all = pd.concat([adj_price_all,adj_price],axis=1)
    pos_trans[contract] = [flag.index]
    
  adj_price_all.columns = raw['contractObject'].unique()
  ret = pd.concat([ret,adj_price_all.reset_index().melt(id_vars = 'index', var_name='cont',value_name=price)],axis=1)
#%%
'''
def f(group):
    s = group['settlePrice']
    first_non_nan_index = s.first_valid_index()
    first_non_nan_value = s[first_non_nan_index]
    group['settleRet'] = s/first_non_nan_value
    for price in ['openPrice', 'highestPrice', 'lowestPrice', 'closePrice']:
        group[price] = group[price]*group['settleRet']
    return group
#%%
ret = pd.merge(raw.loc[raw['mainCon']==1, ['tradeDate', 'contractObject', 'openPrice', 'highestPrice', 'lowestPrice', 'closePrice']], ret, left_on= ['tradeDate', 'contractObject'], right_on = ['index','cont'])
ret = ret.groupby('cont').apply(f)
'''
#%%
result = ret.T.drop_duplicates().T
#result.columns = ['tradeDate','contractObject', 'Open','High', 'Low', 'Close','Settle','SettleRet']
result.to_pickle("ohlcdata.pkl")
#%%

result.loc[result['contractObject']=='A',['tradeDate','Close']].plot(x = 'tradeDate',figsize=(16,10))

raw.loc[raw['contractObject']=='A'&raw['mainCon']==1,['tradeDate','closePrice']].plot(x = 'tradeDate',figsize=(16,10))