#%% By Janis@231021
import requests
import json
import time
import numpy as np
import os
import datetime as dt
import pandas as pd
import ohlc
#%%鉴权
# Token accessToken 及权限校验机制
getAccessTokenUrl = 'https://quantapi.51ifind.com/api/v1/get_access_token'
#获取refresh_token需下载Windows版本接口包解压，打开超级命令-工具-refresh_token查询
refreshtoken = ''#你自己的token

getAccessTokenHeader = {"Content- Type": "application/json", "refresh_token": refreshtoken}

getAccessTokenResponse = requests.post(url=getAccessTokenUrl, headers=getAccessTokenHeader)

accessToken = json.loads(getAccessTokenResponse.content)['data']['access_token']

thsHeaders = {"Content-Type": "application/json", "access_token": accessToken}
#%%
def ths_get_ohlcv(codes, starttime, endtime, interval = "15"):
    thsUrl = 'https://quantapi.51ifind.com/api/v1/high_frequency'

    thsPara =  {
                "codes": codes,
                "indicators": "open,high,low,close,volume",
                "starttime": starttime,
                "endtime": endtime,
                "functionpara": {
                    "Interval": interval,
                    "Fill": "Original",
                    }
                }    
    thsResponse = requests.post(url=thsUrl, json=thsPara, headers=thsHeaders)
    return thsResponse.content

#%%
def ths_json_to_df(response):
    json_object = json.loads(response)
    
    # Create a dictionary to store the data
    data = {}
    
    # Iterate through the JSON object
    for table in json_object["tables"]:
        thscode = table["thscode"]
        time = table["time"]
        open_ = table["table"]["open"]
        high = table["table"]["high"]
        low = table["table"]["low"]
        close = table["table"]["close"]
        volume = table["table"]["volume"]
    
        # Create a dictionary for each row of data
        for i in range(len(time)):
            row = {
                "thscode": thscode,
                "time": time[i],
                "open": open_[i],
                "high": high[i],
                "low": low[i],
                "close": close[i],
                "volume": volume[i]
            }
    
            # Add the row to the data dictionary
            if thscode in data:
                data[thscode].append(row)
            else:
                data[thscode] = [row]
                
    df = pd.DataFrame()
    for table in data:
        print(table)
        df = pd.concat([df, pd.DataFrame(data[table])])
    return df    
#%%
if __name__ == "__main__":
    
    codes = "NIZL.SHF,AGZL.SHF"
    starttime = "2022-10-01 09:00"
    endtime = "2023-10-22 09:00"
    
    resp = ths_get_ohlcv(codes, starttime, endtime)
    print(resp)
    df = ths_json_to_df(resp)
    df.to_excel("NI_AG.xlsx",index=False)













