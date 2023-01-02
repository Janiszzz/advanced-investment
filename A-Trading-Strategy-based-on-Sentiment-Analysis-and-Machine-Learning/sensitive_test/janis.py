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
def single_sort(df, cht, qt):
    #df a dataframe, cht a str of characteristic, qt a list of quantiles
    def f(group):
        group[cht+'_type'] = pd.qcut(group[cht], qt, labels=False, duplicates='drop')
        return group
    df = df.groupby('date').apply(f)
    '''
    l = qt
    mean = np.zeros(l)
    t = np.zeros(l)
    mktadj = np.zeros(l)
    av = np.zeros(l)
    
    for i in range(l):
        temp = df.loc[df[cht+'_type'] == i]
        mean[i] = temp['ret'].mean()
        t[i] = ttest_1samp(temp['ret'].dropna(), 0)[0]
        mktadj[i] = (temp['ret']-temp['mktret']).mean()
        av[i] = (temp[cht]).mean()
    print(mean,'\n',t,'\n',mktadj,'\n',av,'\n')
    '''
    return df
    
def double_sort(df, cht, qt1, qt2):
    #df a dataframe, cht a 2-length str list of characteristic, qt a list of quantiles
    df = single_sort(df, cht[0], qt1)
    def f(group):
        group = single_sort(group, cht[1], qt2)
        return group
    df = df.groupby(cht[0]+'_type').apply(f)
    return df

def print_sort(df, cht):
    df = double_sort(df, cht, 5, 5)
    mean = np.zeros([5,5])
    t = np.zeros([5,5])
    for i in range(5):
        for j in range(5):
            temp = df.loc[((df[cht[0]+'_type'] == i) & (df[cht[1]+'_type'] == j)),'ret']
            mean[i,j] = temp.mean()
            t[i,j] = ttest_1samp(temp.dropna(), 0)[0]
    
    table = pt.PrettyTable([' ','L','2','3','4','H'])        
    mean = np.around(mean,3)
    t = np.around(t,3)
    table.add_row(np.hstack((['L'],mean[0])))
    table.add_row(np.hstack((['t-stat'],t[0])))
    for i in range(1,4):
        table.add_row(np.hstack(([str(i+1)],mean[i])))
        table.add_row(np.hstack((['t-stat'],t[i])))
    
    table.add_row(np.hstack((['H'],mean[4])))
    table.add_row(np.hstack((['t-stat'],t[4])))
    print(table)
    
    return mean,t
    
    
def gen_fct(df, cht, qt1, qt2):
    df = double_sort(df, cht, qt1, qt2)
    def f(group):
        #a for cht0, b for cht1
        #A for fit 'maxa' restrict
        A = (group[cht[0]+'_type'] == len(qt1)-2)
        B = (group[cht[1]+'_type'] == len(qt2)-2)
        a = (group[cht[0]+'_type'] == 0)
        b = (group[cht[1]+'_type'] == 0)
        #M for middle
        MA = ((group[cht[0]+'_type'] != len(qt1)-2)&(group[cht[0]+'_type'] != 0))
        MB = ((group[cht[1]+'_type'] != len(qt2)-2)&(group[cht[1]+'_type'] != 0))
        
        AB = group.loc[A & B,'ret'].mean()
        AMB= group.loc[A & MB,'ret'].mean()
        Ab = group.loc[A & b,'ret'].mean()
        
        aB = group.loc[a & B,'ret'].mean()
        aMB= group.loc[a & MB,'ret'].mean()
        ab = group.loc[a & b,'ret'].mean()
        
        BMA = group.loc[B & MA,'ret'].mean()
        bMA = group.loc[b & MA,'ret'].mean()
        
        AB,AMB,Ab,aB,aMB,ab,BMA,bMA = np.nan_to_num(np.array([AB,AMB,Ab,aB,aMB,ab,BMA,bMA])) 

        group[cht[0]+'_factor'] = (AB+AMB+Ab - aB-aMB-ab)/len(qt2)-2
        group[cht[1]+'_factor'] = (AB+BMA+aB - Ab-bMA-ab)/len(qt1)-2
        
        return group
    df = df.groupby('date').apply(f)
    df = df.drop_duplicates('date')
    df = df.dropna()
    #截至，12个月的和
    #df[cht[0]+'_y'] = df[cht[0]+'_m'].rolling(12).mean()
    #df[cht[1]+'_y'] = df[cht[1]+'_m'].rolling(12).mean()
    return df[['date',cht[0]+'_factor',cht[1]+'_factor']]
#%%
def gen_rol_window(df, idl, l):#uncompleted
    #idl for a id list, l for length of window '3s' '60d', df index must be date
    ret = df.groupby(idl)['ret'].resample(l, on="date").sum()
    df = df.groupby(idl).resample(l, on="date").last()
    df['ret'] = ret
    return df

def fama_macbeth(df, idl, l, fctl, y):
    #fctl for factor list as independent variables  
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
#%%
def add_mktret(df):
    
    def f(group):
        group['mktret'] = group['ret'].mean()
        return group
    df = df.groupby('date').apply(f)
    return df

def downside_beta(df, idl, window):
    #date as index of df, idl for id as Permno, window is length of rolling
    #df = df[['date',idl,'ret','mktret']]
    
    def f1(group, window):
        
        if(group.shape[0]<window):
            return
        def f2(ret,mktret):
            key = np.where(mktret<0)
            ret = ret[key]
            mktret = mktret[key]
            cov = np.cov(np.array([ret,mktret]))
            var = mktret.var()
            return cov[1,0]/var
        group['db'] = rolling_apply(f2, window, group['ret'].values, group['mktret'].values)  
        return group

    df = df.groupby(idl).apply(f1,window)
    return df
#%%

def neu(df,fctl):
    formula = fctl[0] + '~1+' + fctl[1]
    model = smf.ols(formula,df).fit()
    df['resid_'+fctl[0] + '_' + fctl[1]] = model.resid
    print(model.params,'\nt\n',model.tvalues)
    return df
#%%
def gb_vol(df, idl, window):
    #date as index of df, idl for id as Permno, window is length of rolling
    df = df[['date',idl,'ret','mktret']]
    
    def f1(group, window):
        
        if(group.shape[0]<window):
            return None
        def f2(ret,mktret):
            window_mean = mktret.mean()
            key_g = np.where(ret>=window_mean)
            g_vol = ret[key_g].std()
            return g_vol
        def f3(ret,mktret):
            window_mean = mktret.mean()
            key_b = np.where(ret<window_mean)
            b_vol = ret[key_b].std()
            return b_vol    
        
        group['G'] = rolling_apply(f2, window, group['ret'].values, group['mktret'].values)  
        group['B'] = rolling_apply(f3, window, group['ret'].values, group['mktret'].values)  
        return group

    df = df.groupby(idl).apply(f1,window)
    return df








