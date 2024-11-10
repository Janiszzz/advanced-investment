#%% By Janis@231025
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
#%% API list
'''
df = #A Dataframe of single contract: NumericIndex, timestamp, O, H, L, C, V
pic_path = #path where new picture lays

code = #into filename
begin = #numeric timestamp; into filename & period divide
end = #period to the pic
if_train = #bool, for True it labels; False, not label up or down
forecast_length = #period for define label
ma = #moving average window length
'''

'''
Todo:
    修复index排序的问题，先时间index再排序
    增加多品种支持
'''
#%%
def add_ma(df, code_column, ma):
    df['time'] = pd.to_datetime(df['time'])
    df = df.sort_values(by = 'time')

    def f(group,ma):
        group['ma'] = group['close'].rolling(ma).mean()
        return group
    
    df = df.groupby(code_column).apply(f,ma)

    #df['ma'] = df['close'].rolling(ma).mean()
    df = df.reset_index(drop=True)
    return df
   
#%%
def once_generate(df, pic_path, code, begin, end, if_train, forecast_length):
    my_color = mpf.make_marketcolors(volume='white',
                                 ohlc='white')
    my_style = mpf.make_mpf_style(marketcolors=my_color,
                              mavcolors = ['white'],
                              figcolor='(0,0,0)')
    my_config = dict(ohlc_ticksize = 0.3,
                     ohlc_linewidth = 50,
                     volume_width = 0.2,
                     volume_linewidth = 0.2,
                     line_width = 0.4)
    figsize = (4*(end-begin),50)   
    
    df = df.reset_index(drop=True)
    
    if(if_train):             
        df = df.iloc[begin:end+forecast_length+1]
        future_return = df.loc[end+forecast_length]['close']/df.loc[end]['close'] - 1
        #print(future_return,df.loc[end+forecast_length]['close'],df.loc[end]['close'])
        label = '0'
        if(future_return>0.001): label = '1'
        
        name = pic_path + '/' + code + '_' + str(end-begin+1) + '_' + str(forecast_length)  + '_'\
        +  str(df.loc[begin, 'time']).replace(':','-').replace(' ','-') + '_' \
        +  str(df.loc[end, 'time']).replace(':','-').replace(' ','-') + '_' \
        +  str(df.loc[end+forecast_length, 'time']).replace(':','-').replace(' ','-') \
        + '_' +  label + '.png'
        
    elif(not if_train):
        name = pic_path + '/' + code + '_' + str(end-begin+1) + '_' + str(forecast_length)  + '_'\
        +  str(df.loc[begin, 'time']).replace(':','-').replace(' ','-') + '_' \
        +  str(df.loc[end, 'time']).replace(':','-').replace(' ','-') \
        +'.png'
        
        
    ohlc = df.loc[begin:end]
    #ohlc.columns = ['code','Date','Open','High','Low','Close','Volume','ma']
    ohlc = ohlc.rename(
        columns= {
             'time': 'Date',
             'open': 'Open',
             'high': 'High',
             'low': 'Low',
             'close': 'Close',
             'volume': 'Volume',  
         }
        )
    ohlc = ohlc.set_index(['Date'])
    
    ap = mpf.make_addplot(ohlc[['ma']],color='white',width=30)
    
    mpf.plot(ohlc,
             type='ohlc',
             volume=True,
             style=my_style,
             figsize = figsize,
             addplot=ap,
             axisoff=True,
             tight_layout=True,
             scale_padding=0,
             update_width_config = my_config,
             savefig = dict(fname=name,dpi=5,pad_inches=0),
             #returnfig=True,
             closefig = True
             )
    print(name)
    #plt.savefig(name, dpi=300, bbox_inches='tight')

    return [code, end-begin+1, forecast_length \
        , str(df.loc[begin, 'time']).replace(':','-').replace(' ','-') \
        , str(df.loc[end, 'time']).replace(':','-').replace(' ','-') \
        , str(df.loc[end+forecast_length, 'time']).replace(':','-').replace(' ','-') \
        , label, name]

def rolling_generate(df, pic_path, code, begin, end, interval, if_train, forecast_length):
    
    info_list = []
    for i in range(end-interval-forecast_length+1):
        try:
            info = once_generate(df, pic_path, code, i, i+interval-1, if_train, forecast_length)
            info_list.append(info)

        except Exception:  
            print("Except!")
            pass

    return info_list

def multi_generate(df, pic_path, code_column, begin, end, if_train, forecast_length):
    
    def f(code, group):
        once_generate(group, pic_path+"\\"+code, code, begin, end, if_train, forecast_length)
        return
        
    df.groupby(code_column).apply(lambda x: f(x.name, x))
    
    return 0

def multi_rolling_generate(df, pic_path, code_column, begin, end,  interval, if_train, forecast_length):
    
    def f(code, group):
        rolling_generate(group, pic_path+"\\"+code, code, begin, end,  interval, if_train, forecast_length)
        return
        
    df.groupby(code_column).apply(lambda x: f(x.name, x))
    
    return 0





if __name__ == "__main__":
    
    #os.chdir("./data")
    df = pd.read_excel("./data/NI_AG.xlsx")
    df = add_ma(df, "thscode", 5)
    
    #once_generate(df, path, "NIZL.SHF", 5, 21, False, 5)
    calc = df.loc[df['thscode'] == 'NIZL.SHF'].reset_index(drop = True)
    rolling_generate(calc, "./pic", "NIZL.SHF", 0, 10, 5, True, 1)
    
    #multi_generate(df, path, "thscode", 5, 21, True, 5)
    #multi_rolling_generate(df, path, "thscode", 0, df.index[-1], 4, True, 1)




