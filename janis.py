import numpy as np
import matplotlib.pyplot as plt
import cvxopt as opt
from cvxopt import blas, solvers
import pandas as pd
from scipy.stats import ttest_1samp
import prettytable as pt
import datetime as dt
import os
import statsmodels.formula.api as smf
from numpy_ext import rolling_apply
solvers.options['show_progress'] = False
np.set_printoptions(precision=3)
solvers.options['show_progress'] = False
#%%
def add_mkt_ret(df):
    #add a column of median of cross-sectional market returns
    def f(group):
        group['mkt_median'] = group['ret'].median()
        group['mkt_mean'] = group['ret'].mean()
        return group
    df = df.groupby('date').apply(f)
    
    return df

def calc_ret(df, idl, price):
    #idl is PERMNO or windcode etc., price to calc return rate
    def f(group):
        group['ret'] = np.log(group[price]/group[price].shift(1))
        return group
    df = df.groupby(idl).apply(lambda x: x.sort_values('date', ascending=True))
    df = df.reset_index(drop=True)
    df = df.groupby(idl).apply(f)
    df = add_mkt_ret(df)
    return df

def single_sort(df, cht, qt):
    #df a dataframe, cht a STRING of firm characteristic (because this is single sort), qt a list of quantiles.
    
    #in each cross-section, construct portfolios according to quantiles, assign a 'type' id.
    #Notice: qcut is from small to big in value. Smaller the type is, smaller the value is.
    
    def f(group):
        group[cht+'_type'] = pd.qcut(group[cht], qt, labels=False, duplicates='drop')
        return group
    df = df.groupby('date').apply(f)
    
    
    #Print single sort summary chart.
    l = (len(qt)-1)
    #mean and t-stat of each portfolio's returns.
    mean = np.zeros(l+1)
    t = np.zeros(l+1)
    mktadj_mean = np.zeros(l+1)
    mktadj_t = np.zeros(l+1)
    average_factor = np.zeros(l+1)
    q25 = np.zeros(l+1)
    q75 = np.zeros(l+1)

    #In fact, cross-sectional average return can not be calculated by "log return's mean". But I'm lazy.
    #Only one stock's time series return can be calculated by log return's arithmetic mean or sum.
    #To calc cross-sectional average return, you need to multiple each stock's simple return rate.
    
    for i in range(l):
        temp = df.loc[df[cht+'_type'] == i]
        mean[i] = temp['ret'].mean()
        t[i] = ttest_1samp(temp['ret'].dropna(), 0)[0]
        mktadj_mean[i] = (temp['ret'] - temp['mkt_mean']).mean()
        mktadj_t[i] = ttest_1samp((temp['ret'] - temp['mkt_mean']).dropna(), 0)[0]
        average_factor[i] = temp[cht].mean()
        q25[i] = temp['ret'].quantile(0.25)
        q75[i] = temp['ret'].quantile(0.75)

    factor = df.loc[df[cht+'_type'] == l-1].groupby('date').mean()-df.loc[df[cht+'_type'] == 0].groupby('date').mean()
    
    mean[l] = factor['ret'].mean()
    t[l] = ttest_1samp(factor['ret'].dropna(), 0)[0]
    mktadj_mean[l] = (factor['ret'] - factor['mkt_mean']).mean()
    mktadj_t[l] = ttest_1samp((factor['ret'] - factor['mkt_mean']).dropna(), 0)[0]
    average_factor[l] = factor[cht].mean()
    q25[i] = temp['ret'].quantile(0.25)
    q25[l] = factor['ret'].quantile(0.25)
    q75[l] = factor['ret'].quantile(0.75)
    
    
    
    c = {
        'mean':mean,
        't':t,
        'mktadj_mean':mktadj_mean,
        'mktadj_t':mktadj_t,
        'average_factor':average_factor,
        'Q25':q25,
        'Q75':q75,
        #'obs':obs
        }
 
    return df, c
    
def double_sort(df, cht, qt1, qt2):
    #df a dataframe, cht a 2-length str LIST of firm characteristic(because of 'double' sort), qt a list of quantiles
    
    
    df, drop = single_sort(df, cht[0], qt1)
    def f(group):
        group, drop = single_sort(group, cht[1], qt2)
        return group
    df = df.groupby(cht[0]+'_type').apply(f)
    
    #Print double sort summary chart.
    
    #Notice: label is always descending order. When calculate factor like SMB, it actually get BMS!
    #So, you need to reverse the order(or just take a negative value) when you making chart in your paper!
    
    l1 = len(qt1)-1
    l2 = len(qt2)-1
    mean = np.zeros([l1,l2])
    t = np.zeros([l1,l2])
    
    for i in range(l1):
        for j in range(l2):
            temp = df.loc[(df[cht[0]+'_type'] == i)&(df[cht[1]+'_type'] == j)]
            mean[i][j] = temp['ret'].mean()    
            t[i][j] = ttest_1samp(temp['ret'].dropna(), 0)[0]
        
    return df, mean, t

def gen_fct(df, cht, qt1, qt2):
    
    df,drop,drop = double_sort(df, cht, qt1, qt2)
    
    l1 = len(qt1)-1
    l2 = len(qt2)-1
    def f(group):
        mean = np.zeros([l1,l2])
        
        for i in range(l1):
            for j in range(l2):
                temp = group.loc[(group[cht[0]+'_type'] == i)&(group[cht[1]+'_type'] == j)]
                mean[i][j] = temp['ret'].mean() 

        #Always notice the problem noticed in prev section! the long-short portfolio may be short-long in that case! I copy it again:
        #Notice: label is always descending order. When calculate factor like SMB, it actually get BMS!
        #So, you need to reverse the order(or just take a negative value) when you making chart in your paper!
        
        mean_row = np.average(mean, axis=1)
        mean_column = np.average(mean, axis=0)
        
        #lsp for long-short portfolio
        group['lsp_'+cht[0]] = mean_row[l1-1] - mean_row[0]
        group['lsp_'+cht[1]] = mean_column[l2-1] - mean_column[0]
        
        return group
    
    #Notice: FF construct their portfolio each year, and hold for one year. But our func don't realize this. In fact, you can modify source data to apply this method: set the firm characteristic which is used for sort as the same in one year. In that way, for each date in the year, the cross sectional data will output the same sort result, hence same long short portfolios.
    df = df.groupby('date').apply(f)
    
    return df[['date','lsp_'+cht[0],'lsp_'+cht[1]]].drop_duplicates()

#%%
def fama_macbeth(df, fctl, y):
    #fctl a list of independent variables column name, y a string of dependent var name
    #the variables should be of columns of df
    def f(fctl):
        fct_string = ''
        for i in fctl:
            fct_string = fct_string + '+'  + i 
        return fct_string
    
    def ols_coef(x,formula):
        return smf.ols(formula,data=x).fit().params
    def ols_resid(x,formula):
        return smf.ols(formula,data=x).fit().resid

    
    coef_t = df.groupby('date').apply(ols_coef,y+' ~ 1'+ f(fctl))    
    resid_t = df.groupby('date').apply(ols_resid,y+' ~ 1'+ f(fctl))
    df.index.name = 'id'
    resid_t = resid_t.to_frame(name='resid')
    resid_t.index.names=['date','id']
    resid_t = pd.merge(resid_t,df,on=['id','date'])
    resid_t = resid_t.reset_index(drop=True)
    #resid_t = resid_t.drop(columns=['id'])
    
    def fm_summary(df):
        ans = pd.DataFrame()
        for index, p in df.iteritems():
            s = p.describe().T
            s['std_error'] = s['std']/np.sqrt(s['count'])
            s['tstat'] = s['mean']/s['std_error']
            ans = ans.append(s[['mean','std_error','tstat']])
        return ans
    
    return coef_t, fm_summary(coef_t), resid_t

def neu(df,fctl):
    #neutralize:gain residual series
    #we call residual "dependent var neutralized by independent vars".
    formula = fctl[0] + '~1+' + fctl[1]
    model = smf.ols(formula,df).fit()
    df['resid_'+fctl[0] + '_' + fctl[1]] = model.resid
    print(model.params,'\nt\n',model.tvalues)
    return df
#%%
def downside_beta(df, idl, window):
    #downside risk measure β−
    #Ang, Chen, and Xing (2006)
    #date as index of df, idl a string of id name like 'PERMNO', window an int which is length of rolling window

    def f1(group, window):
        
        if(group.shape[0]<window):
            return
        def f2(ret,mkt_median):
            key = np.where(mkt_median<0)
            ret = ret[key]
            mkt_median = mkt_median[key]
            cov = np.cov(np.array([ret,mkt_median]))
            var = mkt_median.var()
            return cov[1,0]/var
        group['db'] = rolling_apply(f2, window, group['ret'].values, group['mkt_median'].values)  
        return group

    df = df.groupby(idl).apply(f1,window)
    return df.reset_index(drop=True)

def gb_vol(df, idl, window):
    #good and bad return volatility
    #Segal, Shaliastovich, and Yaron (2015)
    
    #date as index of df, idl a string of id name like 'PERMNO', window an int which is length of rolling window
    
    def f1(group, window):
        
        if(group.shape[0]<window):
            return None
        def f2(ret,mkt_median):
            window_median = np.median(mkt_median)
            key_g = np.where(ret>=window_median)
            g_vol = ret[key_g].std()
            return g_vol
        def f3(ret,mkt_median):
            window_median = np.median(mkt_median)
            key_b = np.where(ret<window_median)
            b_vol = ret[key_b].std()
            return b_vol    
        
        group['sigma_good'] = rolling_apply(f2, window, group['ret'].values, group['mkt_median'].values)  
        group['sigma_bad'] = rolling_apply(f3, window, group['ret'].values, group['mkt_median'].values)  
        return group

    df = df.groupby(idl).apply(f1,window)
    return df.reset_index(drop=True)








