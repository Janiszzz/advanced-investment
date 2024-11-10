'''
净值序列、净值图，以及各种收益-风险统计数据（年化收益、年化波动、夏普率、最大回撤、收益回撤比，分年度和汇总）
'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt
import os

import plotly.graph_objs as go
from ipywidgets import interact
from ipywidgets import widgets
import plotly.express as px

#%%

    
def strategy_return(ret, weights, fee = [-0.001,-0.005], show_fee = 1):
    #fee 申购 赎回
    if(ret.shape[1] != weights.shape[1]):
        raise Exception("Invalid degree!",ret.shape[1],weights.shape[1])
        
    calc = pd.concat([ret,weights],axis = 1)
    degree = int(calc.shape[1]/2)
    
    calc.iloc[:,degree:] = calc.iloc[:,degree:].fillna(method = 'ffill')
    calc.iloc[:,degree:] = calc.iloc[:,degree:].shift(1)
    calc = calc.dropna()
    
    cost = calc.iloc[:,degree:].diff().fillna(1)
    #第一天是1，diff是后减前，大于零说明权重增加，即申购
    cost[cost>=0] *= fee[0]
    cost[cost<=0] *= fee[1]*-1
    
    calc.columns = np.ones(degree*2)#列名相同，对应才能相乘
    cost.columns = np.ones(degree)
    #收益和权重
    strategy_ret = calc.iloc[:,:degree]*calc.iloc[:,degree:] + cost
    strategy_ret = strategy_ret.sum(axis=1)
    cost = cost.sum(axis=1)
    if(show_fee == 1):
        strategy_ret = pd.concat([strategy_ret, cost], axis = 1)
        strategy_ret.columns = ["strategy_ret","cost"]
    return strategy_ret

#%%策略表现（回测）

def ann_describe(ret):
  
    _ret = ((ret+1).prod()-1)/len(ret)
    ann_ret = _ret*252
    
    vol = ret.std()
    ann_vol = vol*np.sqrt(252)

    return _ret*100,vol*100,ann_ret*100, ann_vol*100

def max_drawdown(rtns):
    max_dd = -1
    peak = rtns[0]
    flag1 = flag2 = peak_flag = 0
    for i in range(0,len(rtns)):
        rtn = rtns[i]
        if rtn > peak:
            peak = rtn
            peak_flag = i
        dd = peak - rtn
        if dd > max_dd:
            max_dd = dd
            flag1 = peak_flag
            flag2 = i
    return max_dd*100, flag1, flag2

def cont_days(x):
    return np.diff(np.where(np.hstack((-1, x, -1)) <= 0)[0]).max() - 1, np.diff(np.where(np.hstack((1, x, 1)) >= 0)[0]).max() - 1

def describe(ret,cost_list):
    # pd.Series ret
    _ret,vol,ann_ret, ann_vol = ann_describe(ret)
    sharp = ann_ret/ann_vol
    
    net_ret = ((ret+1).cumprod())/(ret[0]+1) - 1

    buy = cost_list[[i%2==0 for i in range(len(cost_list))]]
    sell = cost_list[[i%2==1 for i in range(len(cost_list))]]

    buy = net_ret.loc[buy]
    sell = net_ret.loc[sell]

    if(len(sell)<len(buy)):
        sell = pd.concat([sell,pd.Series([ret[-1]])])
        
    spread = sell.reset_index(drop=True) - buy.reset_index(drop=True) 
    
    lose_rate = -1*ret[ret>0].mean()/ret[ret<0].mean()*100   
    win_rate = spread[spread>0].count()/spread.shape[0]*100

    if(ret.name == 'ret'):
         
        win_rate = ret[ret>0].count()/ret.shape[0]*100

    
    positive_days = ret[ret>0].shape[0]/ret.shape[0]*100
    negative_days = ret[ret<0].shape[0]/ret.shape[0]*100

    con_pos_days, con_neg_days = cont_days(ret)
    
    mdd, start, end = max_drawdown((ret+1).cumprod())
    start = ret.index[start]
    end = ret.index[end]
    
    calmar = ann_ret/mdd
    
    data = [ann_ret, ann_vol, sharp, positive_days, negative_days, con_pos_days, con_neg_days, win_rate, lose_rate, mdd, calmar]
    data = [round(x,4) for x in data]
    data = data + [start, end]
    _describe = pd.DataFrame(data).T

    _describe.columns = ['年化收益率', '年化波动率', '夏普率', '盈利天数占比', '亏损天数占比', '最长连续盈利天数', '最长连续亏损天数', '胜率', '盈亏比', '期间最大回撤', '卡玛率', '回撤开始', '回撤结束',]

    return _describe

def describe_cost(s):
    ann_cost = round(len(s[s!=0])/len(s)*250,2)
    #print("年均交易次数：",ann_cost)
    return ann_cost, s[s!=0].index
#%%
def holding(df):
    widgets.SelectionRangeSlider(
        options=df.index,
        description='Dates',
        orientation='horizontal',
        layout={'width': '800px'})
        
        
    @interact
    def read_values(
        slider = widgets.SelectionRangeSlider(
        options = [(i.strftime('%Y%m'), i) for i in df.index],
        index=(0, len(df.index) - 1),
        description='Dates',
        orientation='horizontal',
        layout={'width': '800px'},
        continuous_update=False
    )
    ):
        fig = go.Figure()
        
        period = df.loc[slider[0]:slider[1]]
        costt, cost_list = describe_cost(period['cost'])
        period = period.drop('cost',axis=1)
    
        sheet = pd.DataFrame()
        
        for i in period.columns:
            sheet = pd.concat([sheet,describe(period[i],cost_list)])
        #period.columns = ['基准','最大夏普','最小方差','风险平价','风险预算']
        sheet.index = period.columns
        sheet['年均交易次数'] = costt
        
        from IPython.display import display
        display(sheet)   
        sheet.to_excel("sheet.xlsx")
        
        period = (period+1).cumprod()/(period.iloc[0,:]+1)
        
        for i in period.columns:
            trace = go.Scatter(x=list(period.index), y=list(period.loc[:,i]),name = i)
            fig.add_trace(trace)
        
        fig.update_xaxes(range=[slider[0], slider[1]])
        go.FigureWidget(fig.to_dict()).show()

#%%
if __name__ == "__main__":
    os.chdir(r"D:\Janis\Desktop\大类配置")
    df = pd.read_excel("test.xlsx")
    try:
        df = df.drop('id',axis=1)
    except:
        pass
    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date')
    df.sort_index(ascending=True, inplace=True)
    holding(df)
    
    
    


