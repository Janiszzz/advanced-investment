import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt
import os

plt.rcParams['font.sans-serif']=[u'SimHei']
plt.rcParams['axes.unicode_minus']=False

os.chdir(r"D:\Janis\Desktop\大类配置")
import bar_backtest
import factor
import model

def collect_output(ret, setting):

    weights = model.rolling_re_position_risk_parity(ret,setting['factors'],setting['reposition_days'],setting['modify_level'])
    strategy_ret = bar_backtest.strategy_return(ret, weights)
    return bar_backtest.describe(strategy_ret.iloc[:,0],bar_backtest.describe_cost(strategy_ret.iloc[:,1])[1]),strategy_ret, weights


def read_factor_file(file):
    factors = pd.read_csv(file)
    factors.columns = ['date','signal']
    factors = factors.set_index('date')
    factors.index = pd.DatetimeIndex(factors.index)
    factors = factors.shift(1)
    return factors.dropna()


def digging(ret, setting):
    os.chdir(setting['factor_path'])
    result = pd.DataFrame()  
    for file in os.listdir(setting['factor_path']): 
        try:
            if not file.endswith(".csv"): 
                continue
            setting['factors'] = [read_factor_file(file)]
            temp = collect_output(ret, setting)[0]
            result = pd.concat([result,temp])   
        except:
            continue
    result.index = [file for file in os.listdir(setting['factor_path']) if file.endswith(".csv")]
    return result.sort_values(by = setting['dig_filter'])

def valid(ret,target_factor_files,setting):
    os.chdir(setting['factor_path'])
    result = pd.DataFrame()
    for file in target_factor_files:
        try:
            setting['factors'] = [read_factor_file(file)]
            #print(setting['factors'])
            temp = collect_output(ret, setting)[0]
            #temp = pd.DataFrame(temp)
            result = pd.concat([result,temp])   
        except:
            continue
    result.index = target_factor_files
    return result.sort_values(by = setting['dig_filter'])

def get_ret(ret,target_factor_files,setting):
    os.chdir(setting['factor_path'])
    result_strategy_ret = pd.DataFrame()
    result_weights = pd.DataFrame()
    for file in target_factor_files:
        try:
            print(file)
            setting['factors'] = [read_factor_file(file)]
            #print(setting['factors'])
            temp,strategy_ret, weights = collect_output(ret, setting)
            result_strategy_ret = pd.concat([result_strategy_ret,strategy_ret.iloc[:,0]],axis=1)
            result_weights = pd.concat([result_weights,weights],axis=1)
        except:
            continue    
    result_strategy_ret.columns = target_factor_files
    #result_weights.columns = target_factor_files
    return result_strategy_ret,result_weights

def output(ret,target_factor_files_best,setting):
    
    #result_best_strategy_ret,result_best_weights = get_ret(ret,target_factor_files_best,setting)

    #result_best_strategy_ret.to_excel("result_best_strategy_ret.xlsx")
    #result_best_weights.to_excel("result_best_weights.xlsx")
    
    temp = pd.DataFrame()
    for file in target_factor_files_best: 
        temp = pd.concat([temp,read_factor_file(file)],axis=1)
        
    temp.columns = target_factor_files_best

    temp = temp.fillna(0).mean(axis=1)

    temp.to_csv("composition_factor.csv")

    result_multi_factor_strategy_ret,result_multi_factor_weights = get_ret(ret,['composition_factor.csv'],setting)

    result_multi_factor_strategy_ret.to_excel("result_composition_factor_strategy_ret.xlsx")
    result_multi_factor_weights.to_excel("result_composition_factor_weights.xlsx")
    
    return result_multi_factor_strategy_ret

def print_figure(ret,naive_ret,bm,columns):

    final_ret = pd.concat([bm,naive_ret,ret],axis=1).dropna()
    final_ret.columns = columns
    (final_ret+1).cumprod().plot()
    
def make_reposition_days(ret,start):
    date_range = ret.loc[start:].index
    tuesday_dates = date_range[date_range.dayofweek == 1]
    return tuesday_dates
    
#%%
bm = factor.get_panel_wsd_data(codes = ["931399.CSI"], field = "close", start = "2010-01-01", end = "2024-01-01")
bm = bm.pct_change()
        
#中证500和中证信用债AA+ 1-3年
data = factor.get_panel_wsd_data(codes = ["000905.SH","931191.CSI"], field = "close", start = "2009-01-01", end = "2024-01-01")
ret = data.pct_change().dropna()

#%%
path = r"D:\Janis\Desktop\大类配置\0121\reach_peak"
os.chdir(path)

setting = {
    'factor_path': path,
    'reposition_days':make_reposition_days(ret,dt.datetime(2013,1,1)),
    'modify_level':4,
    'dig_filter': "夏普率"
    }

dig2013_2020 = digging(ret,setting)

dig2013_2020 = dig2013_2020.sort_values(by = "夏普率",ascending = False)
target_factor_files_best = dig2013_2020.index[0:6]
dig2013_2020.to_excel("dig2013_2020.xlsx")
#%%
path = r"D:\Janis\Desktop\大类配置\0121\reach_peak"
os.chdir(path)

setting = {
    'factor_path': path,
    'reposition_days':make_reposition_days(ret,dt.datetime(2013,1,1)),
    'modify_level':4,
    'dig_filter': "夏普率"
    }

dig_mkt_2013_2020 = digging(ret,setting)

dig_mkt_2013_2020 = dig_mkt_2013_2020.sort_values(by = "夏普率",ascending = False)
target_factor_files_best = dig_mkt_2013_2020.index[0:6]
dig2013_2020.to_excel("dig_mkt_2013_2020.xlsx")


#%%

target_factor_files = [
'reach_peak_M0001428_60_0.333333_1.csv',
'reach_peak_G0003376_60_0.333333_-1.csv',
'reach_peak_G1168811_60_0.333333_-1.csv',
]
a_ret = pd.DataFrame()

for file in target_factor_files:
    setting = {
        'factor_path': path,
        'reposition_days':make_reposition_days(ret,dt.datetime(2020,1,1)),
        'modify_level':10,
        'dig_filter': "夏普率"
        }
    a_ret = pd.concat([a_ret,output(ret,[file],setting)],axis=1)
    
    
temp = a_ret.fillna(0)
temp['haha'] = 1
a = bar_backtest.describe(temp.iloc[:,0],bar_backtest.describe_cost(temp['haha'])[1])

target_factor_files = [
'reach_peak_M0001428_60_0.333333_1.csv',
'reach_peak_G0003376_60_0.333333_-1.csv',
'reach_peak_G1168811_60_0.333333_-1.csv',
    ]

setting = {
    'factor_path': path,
    'reposition_days':make_reposition_days(ret,dt.datetime(2013,1,1)),
    'modify_level':10,
    'dig_filter': "夏普率"
    }

abc_ret = output(ret,target_factor_files,setting)

rret,wweig = get_ret(ret,['composition_factor.csv'],setting)
wweig = pd.concat([wweig,data['000905.SH']],axis=1).dropna()
wweig.iloc[:,2] = (1+wweig.iloc[:,2]).cumprod()
wweig.to_excel("weights.xlsx")

temp = abc_ret[dt.datetime(2020,1,1):].fillna(0)
temp['haha'] = 1
abc = bar_backtest.describe(temp.iloc[:,0],bar_backtest.describe_cost(temp['haha'])[1])
#%%


setting = {
    'factor_path': path,
    'reposition_days':make_reposition_days(ret,dt.datetime(2013,1,1)),
    'modify_level':0,
    'dig_filter': "夏普率"
    }

target_factor_files = [
'reach_peak_M0000273_36_0.333333_1.csv',
    ]

naive_ret = output(ret,target_factor_files,setting)


print_figure(a_ret,naive_ret,bm,['基准','风险平价','社零总额','美国制造业利用率','美国经济衰退概率'])

print_figure(abc_ret,ret.iloc[:,1],None,['中证信用债AA指数','多因子风险平价'])




#%%

data = factor.get_panel_wsd_data(codes = ["000905.SH","931191.CSI"], field = "close", start = "2013-01-01", end = "2024-01-01")
ret = data.pct_change().dropna()



temp = abc_ret.fillna(0)
temp['haha'] = 1
abc = bar_backtest.describe(temp.iloc[:,0],bar_backtest.describe_cost(temp['haha'])[1])

temp = pd.DataFrame(ret.iloc[:,1])
temp['haha'] = 1
n = bar_backtest.describe(temp.iloc[:,0],bar_backtest.describe_cost(temp['haha'])[1])


#%%

temp = factor.get_panel_wsd_data(codes = ["000905.SH"], field = "close", start = "2010-01-01", end = "2024-01-01")
temp = temp.pct_change()
temp = pd.concat([temp,naive_ret - abc_ret],axis=1).dropna()
temp.columns = ['中证500','多因子策略超额绝对值']
((temp)+1).cumprod().plot(secondary_y=['多因子策略超额绝对值'])

#%%

bm = factor.get_panel_wsd_data(codes = ["931399.CSI"], field = "close", start = "2013-01-01", end = "2024-01-01")
bm = bm.pct_change()

print_figure(a_ret,naive_ret,bm,['基准','风险平价','因子1'])
print_figure(ab_ret,naive_ret,bm,['基准','风险平价','因子1、5'])
print_figure(abc_ret,naive_ret,bm,['基准','风险平价','因子1、3、5'])



temp = a_ret[dt.datetime(2020,1,1):].fillna(0)
temp['haha'] = 1
bbb = bar_backtest.describe(temp.iloc[:,0],bar_backtest.describe_cost(temp['haha'])[1])


temp = naive_ret[dt.datetime(2020,1,1):].fillna(0)
temp['haha'] = 1
bbb = bar_backtest.describe(temp.iloc[:,0],bar_backtest.describe_cost(temp['haha'])[1])

temp = bm[dt.datetime(2020,1,1):].fillna(0)
temp['haha'] = 1
bbb = bar_backtest.describe(temp.iloc[:,0],bar_backtest.describe_cost(temp['haha'])[1])

#%%
target_factor_files_best = [
'reach_peak_M0001428_60_0.333333_1.csv',
'reach_peak_G0003376_60_0.333333_-1.csv',
'reach_peak_G0002325_60_0.333333_-1.csv',
'reach_peak_G1168811_60_0.333333_-1.csv',
'reach_peak_M0012303_60_0.333333_-1.csv',

 
 ]
#中证500和中证信用债AA+ 1-3年
data = factor.get_panel_wsd_data(codes = ["000905.SH","931191.CSI"], field = "close", start = "2009-01-01", end = "2024-01-01")
ret = data.pct_change().dropna()

setting = {
        'factor_path': path,
        'reposition_days':make_reposition_days(ret,dt.datetime(2013,1,1)),
        'modify_level':10,
        'dig_filter': "夏普率"
        }

for one_target in target_factor_files_best:

    a_ret = output(ret,[one_target],setting)
    temp = a_ret.fillna(0)
    temp['haha'] = 1
    a = bar_backtest.describe(temp.iloc[:,0],bar_backtest.describe_cost(temp['haha'])[1])
    print(a['夏普率'])
