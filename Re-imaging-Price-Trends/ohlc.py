import matplotlib
matplotlib.use('Agg')
import plotly.graph_objects as go
import numpy as np
import os
import datetime as dt
import pandas as pd
import matplotlib.pyplot as plt
import mplfinance as mpf
from mplfinance.original_flavor import candlestick_ohlc
import matplotlib.gridspec as gridspec

import chinese_calendar

my_color = mpf.make_marketcolors(volume='white',
                                 ohlc='white')
my_style = mpf.make_mpf_style(marketcolors=my_color,
                              mavcolors = ['white'],
                              figcolor='(0,0,0)')
my_config = dict(ohlc_ticksize = 0.45,
                 ohlc_linewidth = 1,
                 volume_width = 0,
                 volume_linewidth = 1,
                 line_width = 1)

def get_tradeday(start_str,end_str):
    start = dt.datetime.strptime(start_str, '%Y-%m-%d')
    end = dt.datetime.strptime(end_str, '%Y-%m-%d')

    lst = chinese_calendar.get_workdays(start,end)
    expt = []

    for time in lst:
        if time.isoweekday() == 6 or time.isoweekday() == 7:
            expt.append(time)

    for time in expt:
        lst.remove(time)
    date_list = [item.strftime('%Y-%m-%d') for item in lst]
    return date_list
    
trade_dates = get_tradeday("2016-01-04","2023-01-13")
trade_dates = [dt.datetime.strptime(x,'%Y-%m-%d') for x in trade_dates]

#%%
os.chdir(r"D:/Janis/Desktop/OHLC")
df = pd.read_pickle("ohlcdata.pkl")
df.columns = ['tradeDate','contractObject', 'Open','High', 'Low', 'Close']

#df = df.loc[df['contractObject'].isin(['FU', 'HC', 'I', 'IC', 'IF', 'IH', 'J', 'JD', 'JM','JR', 'L', 'LR', 'M', 'MA', 'NI', 'OI', 'P', 'PB', 'PM', 'PP','RB', 'RI', 'RM', 'RS', 'RU', 'SF', 'SM', 'SN', 'SR', 'T', 'TA','ZC', 'TF', 'V', 'WH', 'WR', 'Y', 'ZN', 'CY', 'AP', 'SC', 'TS','SP', 'EG', 'CJ', 'UR', 'NR', 'RR', 'SS', 'EB', 'SA', 'PG', 'LU','PF', 'BC', 'LH', 'PK', 'IM', 'SI'])]

raw = pd.read_pickle("future_data_all.pkl")
raw = raw[raw['mainCon']==1]
raw = raw[['tradeDate','contractObject','turnoverVol']]
#%%
df = pd.merge(df,raw,on=['tradeDate','contractObject'])
df = df.set_index('tradeDate')
df.index = pd.to_datetime(df.index)
df.columns = ['id', 'Open','High', 'Low', 'Close', 'Volume']
df.iloc[:,1:] = df.iloc[:,1:].astype(np.float32)
#df = df[df.index < dt.datetime(2021,1,1)]
#%%
def f(group):
    group['ma5'] = group['Close'].rolling(5).mean()
    group['ma20'] = group['Close'].rolling(20).mean()
    group['ma60'] = group['Close'].rolling(60).mean()
    #group = group.reindex(trade_dates, fill_value=np.nan)
    return group
df = df.groupby('id', as_index = False).apply(f)
#%%
def print_ohlc(ohlc, step, file_name):
    
    ap = mpf.make_addplot(ohlc[['ma'+str(step)]],color='white',width=1)
    
    my_color = mpf.make_marketcolors(volume='white',
                                     ohlc='white')
    my_style = mpf.make_mpf_style(marketcolors=my_color,
                                  mavcolors = ['white'],
                                  figcolor='(0,0,0)')
    my_config = dict(ohlc_ticksize = 0.45,
                     ohlc_linewidth = 1,
                     volume_width = 0,
                     volume_linewidth = 1,
                     line_width = 1)
    

    plt,ax = mpf.plot(ohlc,
             type='ohlc',
             volume=True,
             style=my_style,
             figsize = (1,1),
             addplot=ap,
             axisoff=True,
             tight_layout=True,
             scale_padding=0,
             update_width_config = my_config,
             #savefig = file_name + ".png",
             #closefig = True             
             )
    plt.savefig(file_name)         
    plt.clf()
    plt.close('all')
    print(name)
#%%

id_list = df['id'].unique()
def refr_wind():
    global free, fig, root, cwin
    import gc
    if fig is not None:
        fgl = plt.get_fignums()
        if len(fgl) > 1:
            plt.close(fgl[0])
            gc.collect()
        #del fig
        fig.clf()
        plt.clf()
        fig = None
        gc.collect()
        #fig = None
    for widget in cwin.winfo_children():
        widget.destroy()
    free = True
    root.title('Empty Chart')
    gc.collect()
    return

for step in [60]:
    #step = 60
    for _id in id_list:
        calc = df[df['id'] == _id]
        calc = calc.dropna()
        date_list = calc.index.unique().to_series().reset_index(drop=True)
        date_list = date_list[step:-step*2-1].reset_index(drop=True)
        for first in date_list.index:
            try:
                first_day = date_list[first]
                last_day = date_list[first+step-1]
                future_day = date_list[first+step*2-1]
                future_return = calc.loc[future_day,'Close']/calc.loc[last_day,'Close'] - 1
                if(future_return>0):
                    label = '1'
                else:
                    label = '0'
                
                name = "D:/Janis/Desktop/OHLC/" + ('train' if last_day<dt.datetime(2021,1,1) else 'test') + "/" + str(step) + "/" + label + "/" + _id+'.'+str(last_day.strftime('%Y-%m-%d'))+'.'+label+'.'+str(step)+ ".png"
                
                
                #print_ohlc(calc.loc[first_day:last_day], step, name)
                ohlc = calc.loc[first_day:last_day]
                ap = mpf.make_addplot(ohlc[['ma'+str(step)]],color='white',width=1)

                mpf.plot(ohlc,
                         type='ohlc',
                         volume=True,
                         style=my_style,
                         figsize = (1,1),
                         addplot=ap,
                         axisoff=True,
                         tight_layout=True,
                         scale_padding=0,
                         update_width_config = my_config,
                         savefig = dict(fname=name,dpi=120,pad_inches=0),
                         #returnfig=True,
                         closefig = True
                         )
                #print(plt.savefig(name, dpi=120, pad_inches=0))
                print(name)
            except:
                pass
 
    
