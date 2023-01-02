#组合优化：每隔40天回看200天，取efficient frontier上min var/max sharp点的权重，用未来40天的ret和等权策略ret作比较
import numpy as np
import matplotlib.pyplot as plt
import cvxopt as opt
from cvxopt import blas, solvers
import pandas as pd

import datetime as dt
import os
solvers.options['show_progress'] = False
#%%收益是正常收益
os.chdir('D:\Janis\Desktop\研一上\资产定价（投资学）\L9')
f1 = pd.read_table('jyc_Alpha001.txt',sep='\t',header=None)
f1 = f1[[0,4]]
f1.columns = ['date','ret1']
f2 = pd.read_table('jyc_Alpha002.txt',sep='\t',header=None)
f2 = f2[[4]]
f2.columns = ['ret2']
f3 = pd.read_table('jyc_Alpha003.txt',sep='\t',header=None)
f3 = f3[[4]]
f3.columns = ['ret3']
raw = pd.concat([f1,f2,f3], axis=1)
raw['date'] = pd.to_datetime(raw['date'],format='%Y%m%d')
raw = raw[raw['date']>dt.datetime(2016,1,1)]
raw = raw.reset_index(drop=True)

raw['ret1'] = raw['ret1'].apply(lambda x: np.log(x+1))
raw['ret2'] = raw['ret2'].apply(lambda x: np.log(x+1))
raw['ret3'] = raw['ret3'].apply(lambda x: np.log(x+1))

#%%https://zhuanlan.zhihu.com/p/25301907
#min variance
def r_w(l_window=0, r_window=239, today=199):
    calc = raw.loc[l_window:r_window]
    #ret for backsee 200 days
    #test for foresee 40 days
    ret = calc[calc.index <= today]
    test = calc[calc.index > today]
    ret = ret[['ret1','ret2','ret3']]
    test = test[['ret1','ret2','ret3']]
    
    cov = np.matrix(ret.cov().values)
    r = np.matrix(ret.mean())
    i = np.matrix(np.ones(3))
    
    #方向跟reference里的反了所以转置反过来
    a = (r).dot(cov.I).dot(r.T).item(0,0)
    b = (r).dot(cov.I).dot(i.T).item(0,0)
    c = i.dot(cov.I).dot(i.T).item(0,0)
    
    u_min = b/c
    lamda = np.matrix([[a,b],[b,c]]).I.dot(np.matrix([u_min,1]).T)
    
    w = lamda.item(0,0)*(cov.I).dot(r.T)+lamda.item(1,0)*(cov.I).dot(i.T)
    
    #计算期望收益
    r_test = np.matrix(test)
    profit = w.T.dot(r_test.T)
    #print(profit)
    return pd.DataFrame(profit.T)

def b_w(i=199):
    test = raw.loc[i+1:i+40]   
    test = test[['ret1','ret2','ret3']]
    w = np.matrix([0.33333,0.33333,0.33333])
    r = np.matrix(test)
    profit = w.dot(r.T)
    
    return pd.DataFrame(profit.T)
#%%max sharp
def r_w_s(l_window=0, r_window=239, today=199):
    calc = raw.loc[l_window:r_window]
    
    ret = calc[calc.index <= today]
    test = calc[calc.index > today]
    ret = ret[['ret1','ret2','ret3']]
    test = test[['ret1','ret2','ret3']]
    
    return_vec = ret.T
    
    def convert_portfolios(portfolios):
        port_list = []
        for portfolio in portfolios:
            temp = np.array(portfolio).T
            port_list.append(temp[0].tolist())
            
        return port_list
    
    
    n = len(return_vec)
    returns = np.asmatrix(return_vec)
        
    N = 100
    mus = [10**(5.0 * t/N - 1.0) for t in range(N)]

    S = opt.matrix(np.cov(returns))  
        
    pbar = opt.matrix(np.mean(returns, axis=1))     
        
    G = -opt.matrix(np.eye(n))
    h = opt.matrix(0.0, (n ,1))
    A = opt.matrix(1.0, (1, n))
    b = opt.matrix(1.0)
        
    portfolios = [solvers.qp(mu*S, -pbar, G, h, A, b)['x']  for mu in mus]
    
    port_list = convert_portfolios(portfolios)

    returns = [blas.dot(pbar, x) for x in portfolios]
    risks = [np.sqrt(blas.dot(x, S*x)) for x in portfolios] 
        
    m1 = np.polyfit(returns, risks, 2)

    x1 = np.sqrt(m1[2] / m1[0])

    wt = solvers.qp(opt.matrix(x1 * S), -pbar, G, h, A, b)['x'] 
    wt = np.matrix(wt)
    
    #np.asarray(wt), returns, risks, port_list
    
    #计算期望收益
    r_test = np.matrix(test)
    profit = wt.T.dot(r_test.T)
    #print(profit)
    return pd.DataFrame(profit.T)

#%%
def output(func):
    results = pd.DataFrame()
    for i in range(199,1613,40):
        #results = pd.concat([results,raw.loc[i+1:i+40, 'date'],func(i-200+1, i+40, i), b_w(i)])
        temp = pd.concat([raw.loc[i+1:i+40, 'date'].reset_index(drop=True), func(i-200+1, i+40, i), b_w(i)], axis=1)
        results = pd.concat([results, temp])
    
    #results = results.T
    results.columns = ['date','r','benchmark']
    
    results['benchmark'] = results['benchmark'].cumsum()
    results['r'] = results['r'].cumsum()
    
    results.plot(figsize=(16,10), x='date',ylabel='Cumulative Return')
    
    return results
#%%
output(r_w)
output(r_w_s)



