import pandas as pd
import os
from scipy import stats
import datetime
from pandas.api.indexers import FixedForwardWindowIndexer
#%%
os.chdir('D:\Janis\Desktop\研一上\资产定价（投资学）\A5')
msf = pd.read_sas('MSF.sas7bdat')
win_data = {}
lose_data = {}
ans = pd.DataFrame({},columns = ['k = 3','k = 6','k = 9','k = 12'],index = ['j = 3,buy','j = 3,buy_t','j = 3,sell','j = 3,sell_t','j = 3,buy-sell','j = 3,buy-sell_t',                                                                     'j = 6,buy','j = 6,buy_t','j = 6,sell','j = 6,sell_t','j = 6,buy-sell','j = 6,buy-sell_t',                                                                      'j = 9,buy','j = 9,buy_t','j = 9,sell','j = 9,sell_t','j = 9,buy-sell','j = 9,buy-sell_t',                                                                      'j = 12,buy','j = 12,buy_t','j = 12,sell','j = 12,sell_t','j = 12,buy-sell','j = 12,buy-sell_t',])
#%%
#筛选
msf_ret = msf[(msf['HEXCD'] == 1)|(msf['HEXCD'] == 2)]
msf_ret = msf_ret[['PERMNO', 'DATE', 'RET']]
#计算是否前j月有数据
msf_ret['in_window'] = 1

#筛选题目要求时间段
msf_ret = msf_ret[(msf_ret['DATE'] > datetime.datetime(1964,11,1))&(msf_ret['DATE'] < datetime.datetime(1989,11,1))]

#计算季度收益和、in_window和。若in_window=3则意味着一整个季度都有数据
#注意DATE标签变成了3 6 9 12 为季度结束期
msf_ret = msf_ret.set_index(['PERMNO','DATE'])
msf_ret = msf_ret.groupby('PERMNO').resample('3M',level=-1).sum()
msf_ret = msf_ret.reset_index()

msf_ret['DATE'] = msf_ret['DATE'] + pd.DateOffset(months=-1)

#%%
def calc(j,k):
    data = msf_ret.copy(deep=True)
    #计算前j和
    data['sum'] = data.groupby('PERMNO')['RET'].rolling(j//3).sum().reset_index(0, drop = True)
    #计算后k和
    data['k_sum'] = data.groupby('PERMNO')['RET'].rolling(FixedForwardWindowIndexer(window_size = k//3+1)).sum().reset_index(0, drop = True) - data['RET']
    #计算前j个月的计数
    data['window_sum'] = data.groupby('PERMNO')['in_window'].rolling(j//3).sum().reset_index(0, drop = True)
    
    #计数小于j的说明有空缺，剔除
    data = data[data['window_sum'] >= j]
    
    #排序
    data = data.groupby('DATE').apply(lambda x: x.sort_values('sum', ascending=True)).reset_index(drop=True)
    win_group = pd.DataFrame()
    lose_group = pd.DataFrame()
    for i in data.groupby('DATE'):
        num = len(i[1])//10
        win = i[1].tail(num)
        lose = i[1].head(num)
        win = win[['PERMNO','DATE','k_sum']]
        win['DATE'] = win['DATE']
        lose = lose[['PERMNO','DATE','k_sum']]
        win_group = win_group.append(win)
        lose_group = lose_group.append(lose)
    
    buy = win_group.groupby('DATE')['k_sum'].mean().to_frame().dropna()
    sell = lose_group.groupby('DATE')['k_sum'].mean().to_frame().dropna()
    b_s = buy-sell
    buy_t = stats.ttest_1samp(buy['k_sum'], 0)
    sell_t = stats.ttest_1samp(sell['k_sum'], 0)
    b_s_t = stats.ttest_1samp(b_s['k_sum'], 0)
    
    ans.loc['j = {},buy'.format(j),'k = {}'.format(k)] = buy['k_sum'].mean()
    
    ans.loc['j = {},buy_t'.format(j),'k = {}'.format(k)] = buy_t[0]
    
    ans.loc['j = {},sell'.format(j),'k = {}'.format(k)] = sell['k_sum'].mean()
    
    ans.loc['j = {},sell_t'.format(j),'k = {}'.format(k)] = sell_t[0]
    
    ans.loc['j = {},buy-sell'.format(j),'k = {}'.format(k)] = b_s['k_sum'].mean()
    
    ans.loc['j = {},buy-sell_t'.format(j),'k = {}'.format(k)] = b_s_t[0]
#%%
for j in [3,6,9,12]:
    for k in [3,6,9,12]:
        calc(j,k)







