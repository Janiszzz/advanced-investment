import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt
import os

from WindPy import w



import sys
def suppress_print(func):
    def wrapper(*args, **kwargs):
        original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')
        result = func(*args, **kwargs)
        sys.stdout = original_stdout
        return result
    return wrapper

@suppress_print
def get_edb_data(code, start = "2023-01-01", end = "2024-01-01"):
    w.start()
    #w.isconnected() 
    data = w.edb(code, start, end, usedf=True)[1]
    data.index = pd.DatetimeIndex(data.index)
    w.close()
    return data

def get_panel_edb_data(codes, start = "2023-01-01", end = "2024-01-01"):
    data = pd.DataFrame()
    for code in codes:
        data = pd.concat([data,get_edb_data(code, start, end)],axis=1)
    data.columns = codes
    return data

@suppress_print
def get_wsd_data(code, field = "close", start = "2023-01-01", end = "2024-01-01"):
    
    w.start()
    data = w.wsd(code, field, start, end, "Currency=CNY;PriceAdj=B", usedf=True)[1]
    data.index = pd.DatetimeIndex(data.index)
    w.close()
    
    return data

def get_panel_wsd_data(codes, field = "close", start = "2023-01-01", end = "2024-01-01"):
    data = pd.DataFrame()
    for code in codes:
        data = pd.concat([data,get_wsd_data(code, field, start, end)],axis=1)
    data.columns = codes
    return data

#因子生成
#示例，随便做的
def pmi(start, end):#日期，传入字符串形如2023-01-02
    df = get_edb_data("M0017126", start, end)
    df.columns = ["PMI"]

    df['3m'] = df.iloc[:,0].rolling(3).mean()
    df['6m'] = df.iloc[:,0].rolling(6).mean()
    df['9m'] = df.iloc[:,0].rolling(9).mean()
    
    df = df.shift(1).dropna()
    
    def signal(group):#涉及滚动窗口（均线突破）的情形
        if(group['3m'].values<group['6m'].values<group['9m'].values): return 1
        else: return 0
        
    df['signal'] = df.groupby(df.index).apply(signal)
    return df['signal']

def erp_and_pe_ttm(start, end):
    ytm_10y = get_edb_data("S0059749",start,end)
    pe_ttm = get_edb_data("Z9897715",start,end)
    
    erp = pd.concat([ytm_10y,pe_ttm],axis = 1).dropna()
    
    erp.index = pd.DatetimeIndex(erp.index)
    erp['signal_value'] = 100/erp.iloc[:,1] - erp.iloc[:,0]
    #erp['signal_value'].plot()
    
    def signal(x):
        if x > 5: return 1
        elif x < 2: return -1
        else: return 0
        
    erp['signal'] = erp['signal_value'].apply(signal)
    
    pe_ttm.index = pd.DatetimeIndex(pe_ttm.index)
    pe_ttm.columns = ['signal_value']
    pe_ttm['rank'] = pe_ttm['signal_value'].rolling(min_periods=1, window=5*250).rank(pct = True)
    
    #pe_ttm['pct'] = full_period['中证800']
    #pe_ttm.loc[pe_ttm.index >dt.datetime(2012,1,1),'CLOSE'].plot()
    
    def signal(x):
        if x > 0.75: return -1
        elif x < 0.25: return 1
        else: return 0
        
    pe_ttm['signal'] = pe_ttm['rank'].apply(signal)
    
    return erp[['signal_value','signal']],pe_ttm[['signal_value','signal']]

def cost_performance_bonus_value(start, end):
    SH50_DVD = get_wsd_data("000016.SH", "dividendyield2", start, end)
    ytm_10y = get_edb_data("S0059749",start,end)
    
    HS300_PE = get_edb_data("M0342074",start,end)
    ytm_10y_GK = get_edb_data("M1004271",start,end)
    
    cost_performance_bonus = pd.concat([SH50_DVD,ytm_10y],axis = 1).dropna()
    cost_performance_value = pd.concat([1/HS300_PE,ytm_10y_GK],axis = 1).dropna()
    
    cost_performance_bonus['signal_value'] = cost_performance_bonus.iloc[:,0]/cost_performance_bonus.iloc[:,1]
    cost_performance_value['signal_value'] = cost_performance_value.iloc[:,0]/cost_performance_value.iloc[:,1]
    
    return cost_performance_bonus['signal_value'],cost_performance_value['signal_value']
    
#事件函数
def reach_peak(s, window, top_percent, peak_type):
    #前top_percent为1，后top_percent为-1，中间为0
    df = pd.DataFrame(s)
    def f(group):
    
        if(group.rank()[-1] > window*(1-top_percent)):
            return 1
        elif(group.rank()[-1] < window*top_percent):
            return -1
        else:
            return 0
        
    df['signal'] = df.iloc[:,0].rolling(window).apply(f)
    df['signal'] *= peak_type
    
    return df['signal']

def mov_average_improve(s, short, long, improve_type):
    #improve_type = 1:短上穿长看多
    df = pd.DataFrame(s)
    df['short'] = df.iloc[:,0].rolling(short).mean()
    df['long'] = df.iloc[:,0].rolling(long).mean()
    df['signal'] = 0 
    df.loc[df['short'] < df['long'], 'signal'] = -1 
    df.loc[df['short'] > df['long'], 'signal'] = 1 
    df['signal'] *= improve_type
    return df['signal']

def create_historical_peak(s, window, peak_type):
    #peak_type = 1:创新高看多
    df = pd.DataFrame(s)
    def f(group):
        if(group.rank()[-1] == window):
            return 1
        elif(group.rank()[-1] == 1):
            return -1
        else:
            return 0
        
    df['signal'] = df.iloc[:,0].rolling(window).apply(f)*peak_type
    
    return df['signal']
    
def generate_factor_files(factors,setting):
    alpha = pd.DataFrame()
    os.chdir(setting['path'])
    os.makedirs(r"./"+setting['event_type'], exist_ok=True)
    os.chdir(r"./"+setting['event_type'])
    
    if(setting['event_type'] == 'create_historical_peak'):
        for factors_name, factors_data in factors.items():
            for l1 in setting['l1']:
                for peak_type in [-1,1]:
                    name = setting['event_type']+"_"+factors_name+"_"+str(l1)+"_"+str(peak_type)
                    a = create_historical_peak(factors_data.dropna().sort_index(),l1,peak_type)
                    a.to_csv(name+".csv")
                    alpha = pd.concat([alpha,a.rename(name)],axis=1)
                    print(name)            
    if(setting['event_type'] == 'mov_average_improve'):
        for factors_name, factors_data in factors.items():
            for l1 in setting['l1']:
                for l2 in setting['l2']:
                    if(l1<l2):
                        for improve_type in [-1,1]:
                            name = setting['event_type']+"_"+factors_name+"_"+str(l1)+"_"+str(l2)+"_"+str(improve_type)
                            a = create_historical_peak(factors_data.dropna().sort_index(),l1,improve_type)
                            a.to_csv(name+".csv")
                            alpha = pd.concat([alpha,a.rename(name)],axis=1)
                            print(name)  
    if(setting['event_type'] == 'reach_peak'):
        for factors_name, factors_data in factors.items():
            for l1 in setting['l1']:
                for l2 in setting['l2']:
                    for peak_type in [-1,1]:
                        name = setting['event_type']+"_"+factors_name+"_"+str(l1)+"_"+str(l2)+"_"+str(peak_type)
                        a = reach_peak(factors_data.dropna().sort_index(),l1,l2,peak_type)
                        #return a 
                        a.to_csv(name+".csv")
                        alpha = pd.concat([alpha,a.rename(name)],axis=1)
                        print(name)            
    return alpha.sort_index()
    
    
    
#%%
if __name__ == "__main__":
    
    factor_list = [
                    'S0048397',
                    'S5100122',
                    'M0000545',
                    'M0000273',
                    'M0017126',
                    'M0000612',
                    'M0001227',
                    'M0001428',
                    'M0000611',
                    'M0012303',
                    'S0113892',
                    'M0001383',
                    'M0001385',
                    'M5525763',
                    'M0009970',
                    'S0073293',
                    'S0073297',
                    'S0073300',
                    'M5786898',
                    'M6388252',
                    'S6018689',
                    'S5713307',
                    'S5133408',
                    'S6124651',
                    'S5123779',
                    'S0181750',
                    'S5713337',
                    'S0008866',
                    'S0110152',
                    'S2726992',
                    'S6126413',
                    'S0000066',
                    'G1168811',
                    'G0005451',
                    'G0002325',
                    'M0041341',
                    'G0003376',
                    'S0105896',
                    'S0105897',
                    'S0105898',
                    'S0105899',
                    'S0105900',
                    'S0200883',
                    'G1168811',
                    'G0002325',
                    'G0003376',

                    ]
    
    #%%for dig
    all_factors = get_panel_wsd_data(factor_list, start = "2000-01-01", end = "2024-01-01")
    
    setting = {
        'path': r"D:\Janis\Desktop\大类配置\0121",
        'l1': [6,9,36,60],
        'l2': [0.333333],
        'event_type': "reach_peak",
        }
    alpha = generate_factor_files(all_factors,setting)
    
    
    
    
    
    #%%
    best_factor_list = [
        'M0001428',
        'G0003376',
        'G0002325',
        'M0000545',
        'M0012303',
        'M0000273',
        ]
    factors = get_panel_wsd_data(best_factor_list, start = "2000-01-01", end = "2024-01-01")
    setting = {
        'path': r"D:\Janis\Desktop\大类配置\0121",
        'l1': [60],
        'l2': [0.333333],
        'event_type': "reach_peak",
        }
    alpha = generate_factor_files(factors,setting)
    
    #%%
    os.chdir(r"D:\Janis\Desktop\大类配置\0121\mkt")
    factors = cost_performance_bonus_value(start = "2000-01-01", end = "2024-01-01")
    factors = pd.concat([factors[0],factors[1]],axis=1)
    factors.columns = ["cost_performance_bonus","cost_performance_value"]
    setting = {
        'path': r"D:\Janis\Desktop\大类配置\0121\mkt",
        'l1': [5*250],
        'l2': [0.333333],
        'event_type': "reach_peak",
        }
    alpha = generate_factor_files(factors,setting)
   
    factors = erp_and_pe_ttm(start = "2000-01-01", end = "2024-01-01")
    temp = pd.concat([factors[0]['signal_value'],factors[1]['signal_value']],axis=1)
    temp.columns = ['erp','pe_ttm']
   
    factors[0]['signal'].to_csv("erp.csv")
    factors[1]['signal'].to_csv("pe_ttm.csv")
    temp.to_excel("erp_pe_ttm.xlsx")
    
    alpha = generate_factor_files(temp,setting)
    
    #%%
    data = get_panel_wsd_data(codes = ["000905.SH"], field = "close", start = "2010-01-01", end = "2024-01-18")
    #all_factors = get_panel_wsd_data(factor_list, start = "2010-01-01", end = "2024-01-01")
    all_factors = get_panel_wsd_data(factor_list, start = "2000-01-01", end = "2024-01-01")
    result = []
    for single_factor_name,single_factor in all_factors.items():
        calc = pd.concat([data,single_factor],axis=1).ffill().dropna()
        calc.iloc[:,0] = calc.iloc[:,0].pct_change()
        for i in range(2013,2024):
            try:
                window = calc[calc.index.year == i]
                ic = window.corr().values[0][1]
                residuals = window.iloc[:,0] - window.iloc[:,1]
                ir = ic / np.std(residuals) 
                result.append([single_factor_name,window.index[-1],ic,ir])   
            except:
                continue
         
    result = pd.DataFrame(result)
    result.columns = ['code','date','ic','ir']
    result.to_excel("icir0125.xlsx")
    #all_factors.corr().to_excel("corr.xlsx")
    
#%%
    
    efficient_factors = ['M0001428','G0003376','G1168811']

    #efficient_factors = list(set(efficient_factors))
    efficient_factors = all_factors.drop_duplicates()
    efficient_factors = efficient_factors.loc[:, ~efficient_factors.columns.duplicated()]
    result = []
    
    for single_factor_name,single_factor in efficient_factors.items():
        calc = pd.concat([data,single_factor],axis=1).ffill().dropna()
        calc.iloc[:,0] = calc.iloc[:,0].pct_change()
        ic = calc.corr().values[0][1]
        residuals = calc.iloc[:,0] - calc.iloc[:,1]
        ir = ic / np.std(residuals) 
        result.append([single_factor_name,ic,ir])   
    
         
    result = pd.DataFrame(result)
    result.columns = ['code','ic','ir']
    result.to_excel("全年度icir0125.xlsx")
    efficient_factors.corr().abs().to_excel("有效corr0125.xlsx")
    #%%
    first_non_nan_index = all_factors.apply(lambda x: x.first_valid_index())
