import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt
import os

plt.rcParams['font.sans-serif']=[u'SimHei']
plt.rcParams['axes.unicode_minus']=False
os.chdir(r"D:\Janis\Desktop\利率债\0117")
#%%
import talib as tb

func_list = [
    tb.CDL2CROWS, 
tb.CDL3BLACKCROWS, 
tb.CDL3INSIDE, 
tb.CDL3LINESTRIKE, 
tb.CDL3OUTSIDE, 
tb.CDL3STARSINSOUTH, 
tb.CDL3WHITESOLDIERS, 
tb.CDLABANDONEDBABY, 
tb.CDLADVANCEBLOCK, 
tb.CDLBELTHOLD, 
tb.CDLBREAKAWAY, 
tb.CDLCLOSINGMARUBOZU, 
tb.CDLCONCEALBABYSWALL, 
tb.CDLCOUNTERATTACK, 
tb.CDLDARKCLOUDCOVER, 
tb.CDLDOJI, 
tb.CDLDOJISTAR, 
tb.CDLDRAGONFLYDOJI, 
tb.CDLENGULFING, 
tb.CDLEVENINGDOJISTAR, 
tb.CDLEVENINGSTAR, 
tb.CDLGAPSIDESIDEWHITE, 
tb.CDLGRAVESTONEDOJI, 
tb.CDLHAMMER, 
tb.CDLHANGINGMAN, 
tb.CDLHARAMI, 
tb.CDLHARAMICROSS, 
tb.CDLHIGHWAVE, 
tb.CDLHIKKAKE, 
tb.CDLHIKKAKEMOD, 
tb.CDLHOMINGPIGEON, 
tb.CDLIDENTICAL3CROWS, 
tb.CDLINNECK, 
tb.CDLINVERTEDHAMMER, 
tb.CDLKICKING, 
tb.CDLKICKINGBYLENGTH, 
tb.CDLLADDERBOTTOM, 
tb.CDLLONGLEGGEDDOJI, 
tb.CDLLONGLINE, 
tb.CDLMARUBOZU, 
tb.CDLMATCHINGLOW, 
tb.CDLMATHOLD, 
tb.CDLMORNINGDOJISTAR, 
tb.CDLMORNINGSTAR, 
tb.CDLONNECK, 
tb.CDLPIERCING, 
tb.CDLRICKSHAWMAN, 
tb.CDLRISEFALL3METHODS, 
tb.CDLSEPARATINGLINES, 
tb.CDLSHOOTINGSTAR, 
tb.CDLSHORTLINE, 
tb.CDLSPINNINGTOP, 
tb.CDLSTALLEDPATTERN, 
tb.CDLSTICKSANDWICH, 
tb.CDLTAKURI, 
tb.CDLTASUKIGAP, 
tb.CDLTHRUSTING, 
tb.CDLTRISTAR, 
tb.CDLUNIQUE3RIVER, 
tb.CDLUPSIDEGAP2CROWS, 
tb.CDLXSIDEGAP3METHODS
]

func_name = [
    'TwoCrows',
'ThreeBlackCrows',
'ThreeInsideUp/Down',
'Three-LineStrike',
'ThreeOutsideUp/Down',
'ThreeStarsInTheSouth',
'ThreeAdvancingWhiteSoldiers',
'AbandonedBaby',
'AdvanceBlock',
'Belt-hold',
'Breakaway',
'ClosingMarubozu',
'ConcealingBabySwallow',
'Counterattack',
'DarkCloudCover',
'Doji',
'DojiStar',
'DragonflyDoji',
'EngulfingPattern',
'EveningDojiStar',
'EveningStar',
'Up/Down-gapside-by-sidewhitelines',
'GravestoneDoji',
'Hammer',
'HangingMan',
'HaramiPattern',
'HaramiCrossPattern',
'High-WaveCandle',
'HikkakePattern',
'ModifiedHikkakePattern',
'HomingPigeon',
'IdenticalThreeCrows',
'In-NeckPattern',
'InvertedHammer',
'Kicking',
'Kicking-bull',
'LadderBottom',
'LongLeggedDoji',
'LongLineCandle',
'Marubozu',
'MatchingLow',
'MatHold',
'MorningDojiStar',
'MorningStar',
'On-NeckPattern',
'PiercingPattern',
'RickshawMan',
'Rising/FallingThreeMethods',
'SeparatingLines',
'ShootingStar',
'ShortLineCandle',
'SpinningTop',
'StalledPattern',
'StickSandwich',
'Takuri',
'TasukiGap',
'ThrustingPattern',
'TristarPattern',
'Unique3River',
'UpsideGapTwoCrows',
'Upside/DownsideGapThreeMethods' 
    ]
'''
func_list = [tb.CDLCLOSINGMARUBOZU, ]
func_name = ['ClosingMarubozu',]
'''
#%%
def get_small_spread_period(file):
    
    data = pd.read_csv(file)
    data['spd'] = data['ask1'] - data['bid1']
    data['spd'] = data['spd'].rolling(20).mean()
    first_position = (data['spd'] <= 0.01).idxmax()
    data = data.loc[first_position:]
    
    code = file.replace(".csv","").split('-')[-1]
    date = file.replace(".csv","").split('-')[:-1]
    year, month, day = map(int, date)
    date = dt.datetime(year, month, day)
    
    return code,date,data

def generate_bars(tick_data, interval):
    bars = []
    for i in range(0, len(tick_data), interval):
        group = tick_data.iloc[i:i+interval]
        if not group.empty:
            bar = {
                'Open': group[['ask1','bid1',]].iloc[0].mean(),
                'High': group[['ask1','bid1',]].max().max(),
                'Low': group[['ask1','bid1',]].min().min(),
                'Close': group[['ask1','bid1',]].iloc[-1].mean(),
                'Volume': group['volume'].sum(),
                'Amount': group['amt'].sum(),
                'Time': group['time'].index[-1]  # 以最后一个Tick的时间作为Bar的时间
            }
            bars.append(bar)
    return pd.DataFrame(bars)

# 模拟交易
def backtest_by_tick(data,setting):
    
    
    capital = setting['capital']
    initial_capital = capital
    position = setting['position']
    signal_series = setting['signal_series']
    bars = setting['bars']
    trade_list = []
    
    for i in range(len(signal_series)-2):
        
        flag = bars.loc[i,'Time']+1
        
        signal = signal_series[i]
        ask_price = data.at[flag, 'ask1']
        bid_price = data.at[flag, 'bid1']
        ask_volume = data.at[flag, 'asize1']
        bid_volume = data.at[flag, 'bsize1']
        
        if signal == 100:  # 买入信号
            trade_volume = min(capital//ask_price, ask_volume//2)
            capital -= trade_volume * ask_price
            position += trade_volume
            trade_list.append([flag,capital,trade_volume,position]) 
        elif signal == -100 and position > 0:  # 卖出信号
            trade_volume = min(bid_volume//2, position)
            capital += trade_volume * bid_price
            position -= trade_volume
            trade_list.append([flag,capital,trade_volume,position])    
    if position > 0:
        bid_price = data.at[len(signal_series),'bid1']
        capital += position * bid_price
        trade_list.append([len(signal_series),capital,position,position])    
    elif position < 0:
        ask_price = data.at[len(signal_series),'ask1']
        capital -= position * ask_price
        trade_list.append([len(signal_series),capital,position,position])    
    
    return capital - initial_capital#trade_list

def generate(data):
    bars_set = {}
    for bar_interval in range(2,21):
        bars_set[bar_interval] = generate_bars(data, bar_interval)
    global func_list
    signal_set = {}
    for bar_interval in bars_set:
        for i in range(len(func_list)):
            temp = bars_set[bar_interval]
            signal_set[str(bar_interval)+"_"+func_name[i]] = func_list[i](temp.Open, temp.High, temp.Low, temp.Close)
    
    return bars_set,signal_set
    
#%%
#signal_short_set = (np.array(signal_set) * -1).tolist(),
# 初始化资金
def batch_processing(file):
    
    code,date,data = get_small_spread_period(file)
    bars_set,signal_set = generate(data)
    result = []
    for signal_series in signal_set:
        setting = {
            'capital' : 1000000,
            'position' : 0,
            'signal_series' : signal_set[signal_series],
            'bars': bars_set[int(signal_series.split("_")[0])],
            }
        #break
        result.append([code,date]+signal_series.split("_")+[backtest_by_tick(data,setting)])
        
    result = pd.DataFrame(result)
    result.columns = ['代码','日期','K线周期','算子','收益率']

    return result
#%%
if __name__ == "__main__":
    os.chdir(r"D:\Janis\Desktop\利率债\历史数据")
    result_set = pd.DataFrame()
    result_set = pd.concat([result_set,batch_processing("2024-1-3-019725.SH.csv")])
    trade = result_set['收益率']
    trade = list(trade)
#%%
    for file in os.listdir(r"D:\Janis\Desktop\利率债\历史数据"):
     
        result_set = pd.concat([result_set,batch_processing(file)])
    
    
    
    #%%

    for code in result_set['代码'].drop_duplicates():
        for operator in result_set['算子'].drop_duplicates():
            for interval in result_set['K线周期'].drop_duplicates():
                temp = result_set.loc[(result_set['代码'] == code) & (result_set['K线周期'] == interval) & (result_set['算子'] == operator), ['日期','收益率']]
                temp = temp.set_index('日期')
                temp.index = pd.DatetimeIndex(temp.index)
                temp = temp.sort_index()
                cret = temp.mean().sum() 
                if(cret> 0):
                    print(code,operator,interval)
                    print(cret)
                #(temp+1).cumprod().plot()
    














