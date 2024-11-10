import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt
import os
import riskfolio as rp

plt.rcParams['font.sans-serif']=[u'SimHei']
plt.rcParams['axes.unicode_minus']=False
#%%
def risk_parity(ret, modify_matrix = None, risk_budgets = None):
    
    port = rp.Portfolio(returns=ret)
    
    method_mu='hist'
    method_cov='hist'
      
    model = 'Classic'
    rm = 'MV' 
    hist = True 
    rf = 0 
   
    port.assets_stats(method_mu=method_mu, method_cov=method_cov, d=0.94)
    
    if(modify_matrix.any()):
        
        modify_matrix = pd.DataFrame(modify_matrix, index = port.cov.index, columns = port.cov.columns)
        port.cov = modify_matrix.dot(port.cov.dot(modify_matrix.T))
    
    b = risk_budgets

    w = port.rp_optimization(model=model, rm=rm, b=b, rf=rf, hist=hist)
    
    return w.T

def naive_optimize(ret, modify_matrix = None, obj = 'Sharpe'):

    port = rp.Portfolio(returns=ret)
    
    method_mu='hist' 
    method_cov='hist' 
    
    model='Classic' 
    rm = 'MV' 
    hist = True 
    rf = 0 
    l = 0 
    
    port.assets_stats(method_mu=method_mu, method_cov=method_cov, d=0.94)
    
    if(modify_matrix.any()):
        modify_matrix = pd.DataFrame(modify_matrix, index = port.cov.index, columns = port.cov.columns)
        
        port.cov = modify_matrix.dot(port.cov.dot(modify_matrix.T))
    
    w = port.optimization(model=model, rm=rm, obj=obj, rf=rf, l=l, hist=hist)
    
    return w.T
#%%
def get_modify_matrix(a, modify_level):
    
    if(modify_level == 0): return np.diag([1.,1.])
    
    if(a > 0):
        U = np.diag([np.power(np.sqrt(1/modify_level),a),1.])
    elif(a == 0):
        U = np.diag([1.,1.])
    elif(a < 0):
        U = np.diag([np.power(np.sqrt(modify_level),abs(a)),1.])
    else:
        raise Exception("Check signal")
    
    return U
    
def rolling_re_position_risk_parity(ret, factors, reposition_days, modify_level = 0):
    
    weights_list = pd.DataFrame()
    if((not all(pd.api.types.is_datetime64_any_dtype(df.index) for df in factors)) or (not pd.api.types.is_datetime64_any_dtype(ret.index))):
        raise Exception("factors/ret index is not datetime index")
    #检查下是否能对齐
    signal = pd.concat([ret]+factors,axis=1).iloc[:,ret.shape[1]:].ffill().sum(axis=1)[reposition_days].shift(1).fillna(0)
      
    for key_day in reposition_days:
        #print(signal[key_day])
        U = get_modify_matrix(signal[key_day],modify_level)
        
        w = risk_parity(ret = ret.loc[key_day-dt.timedelta(3*250):key_day], modify_matrix = U)    
        weights_list = pd.concat([weights_list,w])
        
    weights_list.index = reposition_days
    #print(signal)
    return weights_list

#%%
if __name__ == "__main__":
    
    os.chdir(r"D:\Janis\Desktop\大类配置")
    
    ret = pd.read_excel("行情序列.xlsx")
    ret = ret.pivot(values = "收盘价(元)", index = "时间", columns = "简称").iloc[:,0:2].pct_change()
    s = np.diag([2,2])
    w = risk_parity(ret = ret.dropna(), modify_matrix = s)
    naive_optimize(ret = ret.dropna(), modify_matrix = s)
    









