import os
import pandas as pd

def classification(path):
    df = os.listdir(path)
    df = [i.split(".png")[0].split("_") for i in df]
    df = pd.DataFrame(df)
    file_list = path+pd.DataFrame(os.listdir(path))
    df = pd.concat([df,file_list],axis=1)
    df.columns = ['code','window_length','forecast_length','start','end','forcast','label','file_name']
    return df.astype(str)

    
    
#%%
if __name__ == "__main__":    
    path = "./pic/"
    df = classification(path)
    df_list =  [group for key, group in df.groupby('code')]
    