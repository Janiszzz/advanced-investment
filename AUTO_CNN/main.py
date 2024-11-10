import os
import ohlc
import ths
import push
import datetime as dt
import pandas as pd
#%%
'''
codes = ['APZL.CZC','CFZL.CZC','CJZL.CZC','CYZL.CZC','FGZL.CZC','JRZL.CZC','LRZL.CZC','MAZL.CZC','OIZL.CZC','PFZL.CZC','PKZL.CZC','PMZL.CZC','PXZL.CZC','RIZL.CZC','RMZL.CZC','RSZL.CZC','SAZL.CZC','SFZL.CZC','SHZL.CZC','SMZL.CZC','SRZL.CZC','TAZL.CZC','URZL.CZC','WHZL.CZC','ZCZL.CZC','AZL.DCE','BBZL.DCE','BZL.DCE','CSZL.DCE','CZL.DCE','EBZL.DCE','EGZL.DCE','FBZL.DCE','IZL.DCE','JDZL.DCE','JMZL.DCE','JZL.DCE','LHZL.DCE','LZL.DCE','MZL.DCE','PGZL.DCE','PPZL.DCE','PZL.DCE','RRZL.DCE','VZL.DCE','YZL.DCE','LCZL.GFE','SIZL.GFE','AGZL.SHF','ALZL.SHF','AOZL.SHF','AUZL.SHF','BCZL.SHF','BRZL.SHF','BUZL.SHF','CUZL.SHF','ECZL.SHF','FUZL.SHF','HCZL.SHF','LUZL.SHF','NIZL.SHF','NRZL.SHF','PBZL.SHF','RBZL.SHF','RUZL.SHF','SCZL.SHF','SNZL.SHF','SPZL.SHF','SSZL.SHF','WRZL.SHF','ZNZL.SHF']
times = ["2016-10-01 09:00","2017-10-01 09:00","2018-10-01 09:00","2019-10-01 09:00","2020-10-01 09:00","2021-10-01 09:00","2022-10-01 09:00","2023-10-01 09:00"]

df = pd.DataFrame()
for code in codes:
    for i in range(len(times)-1):
        try:
            resp = ths.ths_get_ohlcv(code, times[i], times[i+1], interval = "15")
            resp = ths.ths_json_to_df(resp)
            df = pd.concat([df,resp])
        except:
            df.to_pickle("./data/futuredata.pkl")

df.to_pickle("./data/futuredata.pkl")
'''
df = pd.read_pickle("futuredata.pkl")
resp = ths.ths_get_ohlcv('ZNZL.SHF', "2016-10-01 09:00","2017-10-01 09:00", interval = "15")


#%%



%%
import datetime

current_time = datetime.datetime.now()
formatted_time = current_time.strftime("%Y-%m-%d %H:%M:%S")
print("当前时间：", formatted_time)
