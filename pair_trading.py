import numpy as np
import pandas as pd
# import pandas_datareader.data as web
import matplotlib.pyplot as plt
import arch.unitroot as at
import os
import statsmodels
from statsmodels.tsa.stattools import coint
import statsmodels.regression.linear_model as rg
import matplotlib.pyplot as plt
from itertools import combinations

#%%计算spread
def calc_pair_ratio(y,x):
    return rg.OLS(y,x).fit().params[0]

def calc_spread(tset,fset,kind):
    spread = pd.DataFrame()
    asset_list = tset.keys()
    asset_list = combinations(asset_list,2)
    if(kind == 'train'):
        for i in asset_list:
            temp = tset[i[0]] - calc_pair_ratio(tset[i[0]],tset[i[1]]) * tset[i[1]]
            spread = pd.concat([spread,temp.to_frame(name=i[0]+'-'+i[1])],axis=1)
    if(kind == 'forecast'):    
        for i in asset_list:
            temp = fset[i[0]] - calc_pair_ratio(tset[i[0]],tset[i[1]]) * fset[i[1]]
            spread = pd.concat([spread,temp.to_frame(name=i[0]+'-'+i[1])],axis=1)    
    return spread 

#根据spread ADF检验p值筛选最协整的k组
def choose_coint(spread,k):
    p = pd.Series()
    for name,i in spread.iteritems():
        p[name] = at.ADF(i, trend='ct').pvalue
    spread = spread[p.sort_values()[0:k].index] 
    return spread.keys()

#以下进入forcast
#根据zscore和滞后zscore得到trade signal
   
def add_zscore(pair,spread):
    zscore = (spread - spread.rolling(window=21).mean()) / spread.rolling(window=21).std()
    pair.insert(len(pair.columns), 'z-score', zscore)
    pair.insert(len(pair.columns), 'z-score(-1)', zscore.shift(1))
    pair.insert(len(pair.columns), 'z-score(-2)', zscore.shift(2))
    return pair
    
def add_signal(pair):

    #划分为-2~2五种signal
    pair['signal'] = 0
    cdt1 = (pair['z-score(-2)']>-2) & (pair['z-score(-1)']<-2)
    cdt2 = (pair['z-score(-2)']<-1) & (pair['z-score(-1)']>-1)
    cdt3 = (pair['z-score(-2)']< 2) & (pair['z-score(-1)']> 2)
    cdt4 = (pair['z-score(-2)']> 1) & (pair['z-score(-1)']< 1)
    
    pair.loc[cdt1,'signal'] = -2
    pair.loc[cdt2,'signal'] = -1
    pair.loc[cdt3,'signal'] =  2
    pair.loc[cdt4,'signal'] =  1
    
    return pair

def add_pos(pair):
    #根据signal得出position
    #简化变换
    pair['position'] = 0
    pair['position'] = (abs(pair['signal'])-1)*np.sign(pair['signal'])*(-1)
    
    pair.loc[pair['signal']==0,'position'] = None
    pair = pair.fillna(method='ffill')
    
    return pair

def add_fee(pair):
    pair.insert(len(pair.columns), 'position(-1)', pair['position'].shift(1))
    pair['fee'] = 0.0
  
    cdt = (pair['signal']!=0)&(pair['position']!=pair['position(-1)'])   
    pair.loc[cdt,'fee'] =0.001

    return pair

def add_spread_ret(pair,pair_name_list):
    #generate spread of pct-return-form of forecast data
    pair['spread-ret'] = calc_spread(training_raw[pair_name_list],fret[pair_name_list],'forecast')
    return pair
    
def add_strat_ret(pair):
    #pair = pair.dropna()
    
    pair['ret'] = pair['spread-ret'] * pair['position']
    pair['ret-fee'] = pair['ret'] - pair['fee']
    return pair

def add_annal_ret(pair):
    #expect pct return, not raw or log ret
    pair['cumret'] = np.cumprod(pair['ret'] + 1) ** (252 / len(pair)) - 1
    pair['cumret-fee'] = np.cumprod(pair['ret-fee'] + 1) ** (252 / len(pair)) - 1
    pair['cum'+pair.columns[0]] = np.cumprod(pair[pair.columns[0]] + 1) ** (252 / len(pair)) - 1
    pair['cum'+pair.columns[1]] = np.cumprod(pair[pair.columns[1]] + 1) ** (252 / len(pair)) - 1
    return pair

def add_all(df,spread,pair_name):
    
    pair_name_list = pair_name.split('-')
    pair = pd.concat([df[pair_name_list],spread[pair_name].to_frame(name='spread')],axis=1)
    pair = add_zscore(pair, pair['spread'])
    pair = add_signal(pair)
    pair = add_pos(pair)
    pair = add_fee(pair)
    pair = add_spread_ret(pair, pair_name_list)
    pair = add_strat_ret(pair)
    pair = add_annal_ret(pair)
    #print(pair)
    return pair


#%%
def print_result(pair,pair_name):
    i,j = pair_name.split('-')
    
    result = [{'0': 'Annualized:', '1': 'ret', '2': 'ret-c', '3': 'ret '+i, '4': 'ret '+j},
            {'0': 'Return', '1': np.round(pair.iloc[-1]['cumret'], 4),
             '2':np.round(pair.iloc[-1]['cumret-fee'], 4),
             '3': np.round(pair.iloc[-1]['cum'+i], 4),
             '4': np.round(pair.iloc[-1]['cum'+j], 4)},
            {'0': 'Standard Deviation', '1': np.round(np.std(pair['ret']) * np.sqrt(252), 4),
             '2': np.round(np.std(pair['ret-fee']) * np.sqrt(252), 4),
             '3': np.round(np.std(pair[i]) * np.sqrt(252), 4),
             '4': np.round(np.std(pair[j]) * np.sqrt(252), 4)},
            {'0': 'Sharpe Ratio (Rf=0%)', '1': np.round(pair.iloc[-1]['cumret'] / (np.std(pair['ret']) * np.sqrt(252)), 4),
             '2': np.round(pair.iloc[-1]['cumret-fee'] / (np.std(pair['ret-fee']) * np.sqrt(252)), 4),
             '3': np.round(pair.iloc[-1]['cum'+i] / (np.std(pair[i]) * np.sqrt(252)), 4),
             '4': np.round(pair.iloc[-1]['cum'+j] / (np.std(pair[j]) * np.sqrt(252)), 4)}]
    table = pd.DataFrame(result)
    print('')
    print('== '+pair_name.upper()+' Strategy Performace Summary ==')
    print('')
    print(table)


#%%
def figure(pair,pair_name):
    
    i,j = pair_name.split('-')
    
    fig1, ax1 = plt.subplots()
    ax1.plot(pair[i])
    ax1.legend(loc='lower left')
    ax2 = ax1.twinx()
    ax2.plot(pair[j], color='orange')
    ax2.legend(loc='lower right')
    plt.suptitle(pair_name.upper()+' Prices')
    plt.show()
    plt.clf()
    
    fig2, ax = plt.subplots()
    ax.plot(pair['spread'], label='spread')
    ax.axhline(pair['spread'].mean(), color='orange')
    ax.legend(loc='upper left')
    plt.suptitle(pair_name.upper()+' Spread')
    plt.show()
    plt.clf()
    
    fig3, ax = plt.subplots()
    ax.plot(pair['z-score'], label='z-score')
    ax.axhline((-2), color='green')
    ax.axhline((-1), color='green', linestyle='--')
    ax.axhline((2), color='red')
    ax.axhline((1), color='red', linestyle='--')
    ax.legend(loc='upper left')
    plt.suptitle(pair_name.upper()+' Rolling Spread Z-Score')
    plt.show()
    plt.clf()
    
    plt.plot(pair['cumret'], label='Returns')
    plt.plot(pair['cumret-fee'], label='Returns With Trading Commissions ')
    plt.plot(pair['cum'+i], label=i.upper()+' Returns')
    plt.plot(pair['cum'+j], label=j.upper()+' Returns')
    plt.title(pair_name.upper()+' Trading Strategy Cumulative Returns')
    plt.legend(loc='upper left')
    plt.show()
    plt.clf()
    
#%%   
def pair_trading(raw,traintime,forcasttime,k):
    global training_raw
    training_raw = raw[:pd.to_datetime(traintime)]
    forecast_raw = raw[pd.to_datetime(forcasttime):]
    
    tspread = calc_spread(training_raw, forecast_raw, 'train')   
    spread_best = choose_coint(tspread,k)
    global fret
    tret = training_raw.pct_change(1).dropna()
    fret = forecast_raw.pct_change(1).dropna()
    
    fret_spread = calc_spread(training_raw, forecast_raw, 'forecast')
    
    for pair_name in spread_best:
        pair = add_all(fret,fret_spread,pair_name)
        print_result(pair,pair_name) 
        figure(pair,pair_name)
#%%
os.chdir('D:\Janis\Desktop\研一上\资产定价（投资学）\midterm')
raw = pd.read_csv('Pairs-Trading-Analysis-Data.txt')
raw = raw.rename(columns = {'Date':'date'})
raw['date'] = pd.to_datetime(raw['date'])
raw = raw.set_index('date')
raw = raw.dropna()

pair_trading(raw, '2014-12-31', '2015-01-02', 1)











