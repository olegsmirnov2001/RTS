from datetime import datetime, timedelta
import pandas as pd
import requests
import os
import time

market=24
em=18948
code="CL"

period = 3 # 8 - daily, 3 - 5min
period_name = "_5min"

dates_f = ["2019-01-01",
           "2019-04-01",
           "2019-07-01",
           "2019-10-01",
           "2020-01-01",
           "2020-04-01",
           "2020-07-01",
           "2020-08-01"]

urls = []
filenames = []

for i in range(len(dates_f) - 1):
    date_f = datetime.strptime(dates_f[i  ], "%Y-%m-%d")
    date_t = datetime.strptime(dates_f[i+1], "%Y-%m-%d") - timedelta(days=1)
    
    dataname = code + period_name \
        + "_" + date_f.strftime("%Y%m%d") \
        + "_" + date_t.strftime("%Y%m%d")
    
    url = "http://export.finam.ru/export9.out?market=" + str(market) \
            + "&em=" + str(em) \
            + "&code=" + code \
            + "&apply=0&df=" + str(date_f.day) \
            + "&mf=" + str(date_f.month - 1) \
            + "&yf=" + str(date_f.year) \
            + "&from=" + date_f.strftime("%d.%m.%Y")\
            + "&dt=" + str(date_t.day) \
            + "&mt=" + str(date_t.month - 1) \
            + "&yt=" + str(date_t.year) \
            + "&to=" + date_t.strftime("%d.%m.%Y")\
            + "&p=" + str(period) \
            + "&f=" + dataname \
            + "&e=.csv&cn=" + code \
            + "&dtf=1&tmf=1&MSOR=1&mstime=on&mstimever=1" \
            + "&sep=3&sep2=1&datf=1&at=1"
    
    urls.append(url)
    filenames.append("data\\" + dataname + ".csv")

for i in range(len(urls)):
    params={'User-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/84.0.4147.135 Safari/537.36 OPR/70.0.3728.189 (Edition Yx 03)'}
    
    if i > 0:
        time.sleep(1)
    data = requests.get(urls[i],headers=params)
    
    print("Downloaded %d/%d part" % (i + 1, len(urls)))
    
    file_data = open(filenames[i], "wb")
    file_data.write(data.content)
    file_data.close()

file = open("data\\" + code + period_name + ".csv", "wb")
for i in range(len(filenames)):
    with open(filenames[i], "rb") as file_data:
        if i > 0:
            next(file_data)
        data = file_data.read()
    
    file.write(data)
    
    os.remove(filenames[i])

file.close()

print("Merged file is ready")

df = pd.read_csv ("data\\" + code + period_name + ".csv",
                  sep = ';',
                  usecols = ['<DATE>', '<TIME>', '<CLOSE>'],
                  dtype = {'<DATE>': int, '<TIME>': int, '<CLOSE>': float})

df.columns = ['DATE', 'TIME', 'CLOSE']

data_open =df[df.TIME ==  95000]
data_close=df[df.TIME == 163500]

data_merge = (pd.merge(data_open, data_close, on = 'DATE', how = 'inner'))\
                    .drop(['TIME_x', 'TIME_y'], axis = 'columns')

data_merge.columns = ['DATE', 'OPEN', 'CLOSE']

data_merge.DATE = data_merge.DATE.map( \
    lambda date: (datetime.strptime(str(date), "%Y%m%d")).strftime("%Y-%m-%d"))

data_merge.to_csv("data\\" + code + ".csv", sep=';', index=False)

print("Intraday data is ready")