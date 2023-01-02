import bs4, os, requests, re, random
import pandas as pd
from lxml import etree
import json
import time
max_page  = 100
results = pd.DataFrame()
os.chdir(r'D:\Janis\Desktop\研一上\大数据\final\data')

no_list =[]
stock_id_list = pd.read_pickle('hulist.pkl')

#%%
for stock_id in stock_id_list[0]:
    '''
    if (stock_id <= 603600):
        continue
    '''
    time.sleep(5)
    print(stock_id)
    
    proxy = requests.get('your own proxy')
    proxies = {
            'http': 'http://'+proxy.text,
            'https': 'http://'+proxy.text
        }
    try:
        for page in range(1,max_page+1):
            #page=1
            print(str(stock_id)+'+'+str(page))
            title_list=[]
            comment_list=[]
            read_list=[]
            time_list=[]
            age_list=[]
            new_time_list = []
            new_title_list = []
            
            stock_url = f'http://guba.eastmoney.com/list,{stock_id}_{page}.html'
            response  = requests.get(stock_url, proxies=proxies, timeout=5 )
            root = etree.HTML(response.text)
            
            title_list = root.xpath("//span[@class='l3 a3']/a/text()")
            title_list = list(map(str,title_list))  
            comment_list = root.xpath("//span[@class='l2 a2']/text()")
            comment_list = list(map(str,comment_list))
            read_list = root.xpath("//span[@class='l1 a1']/text()")
            read_list = list(map(str,read_list))
            time_list = root.xpath("//span[@class='l5 a5']/text()")
            time_list = list(map(str,time_list))
            
            try:
                comment_list.remove("评论")
                read_list.remove("阅读")
                time_list.remove("最后更新")
            except:
                None
                
            result = {
                    'time'   : time_list[0:80],
                    'title'  : title_list[0:80],
                    'read'   : read_list[0:80],
                    'comment': comment_list[0:80]
                    }
            df = pd.DataFrame(result)
            df['stkcd'] = stock_id
            results = pd.concat([results,df])
    except:
        no_list.append(stock_id)
        continue
  
results.to_pickle('guba.pkl')    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    