import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import math

#==================================================

def pull_data(fname,
              delimiter = ';',
              skip_header = 0,
              missing_values = '',
              filling_values = 0,
              data_cols = None,
              dates_col = 0,
              dates_align = None,
              use_yesterday = False):
    data = np.genfromtxt(fname,
                         delimiter=delimiter,
                         skip_header=skip_header,
                         missing_values=missing_values,
                         filling_values=filling_values,
                         usecols=data_cols)
    
    if dates_align is not None:
        dates_data = np.genfromtxt(fname,
                                   delimiter=delimiter,
                                   dtype=str,
                                   skip_header=skip_header,
                                   usecols=dates_col)
        
        data_aligned = np.empty(dates_align.size if len(data.shape) == 1
                                else (dates_align.size, data.shape[1]),
                                float)
        
        for i in range(dates_align.size):
            if len(np.where(dates_data == dates_align[i])[0]) > 0:
                data_aligned[i] = data[
                        np.where(dates_data == dates_align[i])[0][0]]
            elif use_yesterday and i > 0:
                # print("Missing values! file %s, date %s, used yesterday" \
                      # % (fname, dates_align[i]))
                data_aligned[i] = data_aligned[i-1]
            else:
                # print("Missing values! file %s, date %s" \
                      # % (fname, dates_align[i]))
                data_aligned[i] = filling_values if len(data.shape) == 1 \
                                else np.full(data.shape[1], filling_values)
                # print("filled with " + str(data_aligned[i]))
        
        return data_aligned
    
    return data

def pull_intraday_data(fname,
                       dates_align,
                       delimiter = ';',
                       skip_header = 1,
                       open_col = 1,
                       close_col = 2,
                       dates_col = 0):
    data_open = pull_data(fname,
                          delimiter,
                          skip_header,
                          '',
                          None,
                          open_col,
                          dates_col,
                          dates_align)
    data_close = pull_data(fname,
                           delimiter,
                           skip_header,
                           '',
                           None,
                           close_col,
                           dates_col,
                           np.concatenate([[(datetime.strptime(dates_align[0],
                                    "%Y-%m-%d") - timedelta(days=1)).strftime("%Y-%m-%d")],
                                            dates_align]))
    
    data = np.empty(dates_align.size, float)
    for i in range(len(dates_align)):
        if not math.isnan(data_open[i]) and not math.isnan(data_close[i]):
            data[i] = data_open[i] / data_close[i] - 1
            # print(data[i])
        else:
            # print("Detected something strange!")
            opening = None
            for j in range(i, len(dates_align)):
                if not math.isnan(data_open[j]):
                    opening = data_open[j]
                    break
            else:
                print("Missing last value! date=%s" % dates_align[i])
                data[i]=0
                continue
            closing = None
            for k in range(i, -1, -1):
                if not math.isnan(data_close[k]):
                    closing = data_close[k]
                    break
            else:
                print("Missing first value! date=%s" % dates_align[i])
                data[i]=0
                continue
            # print("Replaced with %d-gap" % (j - k))
            data[i] = (opening / closing - 1) / (1 + j - k)
            # print(data[i])
    
    return data

#==================================================

dates = np.genfromtxt("data\\IMOEX_1-5.csv",
                      delimiter = ';',
                      dtype=str,
                      skip_header=1,
                      usecols=3)

imoex = pull_data("data\\IMOEX_1-5.csv",
                     skip_header=1,
                     data_cols=4)

sz = imoex.size

data = pull_data("data\\securities_moex.csv",
                 skip_header=2,
                 data_cols=range(78, 116),
                 dates_col=77,
                 dates_align=dates)
data = pd.DataFrame(data)

names = ['GAZP','SBER','LKOH','GMKN','YNDX','NVTK','ROSN','TATN','SNGS','PLZL','MGNT','MTSS','POLY','FIVE','SNGSP','IRAO','MOEX','ALRS','NLMK','TCSG','CHMF','SBERP','VTBR','PHOR','TATNP','TRNFP','MAGN','RTKM','RUAL','HYDR','AFKS','PIKK','FEES','DSKY','AFLT','CBOM','LSRG','UPRO']
data.columns = names
# Made throw asshole

#==================================================

orderings = pd.read_csv("data\\ordered_securities.csv", sep=';')
print(orderings)

for method in orderings.columns:
    order = orderings[method]
    for i in range(len(order), 1, -1):
        reg = LinearRegression().fit(data[order[:i]], imoex)
        print(pd.Series(reg.coef_, index=order[:i]))

names = names[(-weights).argsort()].tolist()
data.columns = names

print(data)
print(names)

#==================================================

coefs = []
scores = []
for code in sec_names_sorted:
    reg = LinearRegression().fit(data, rts)
    coefs.append(reg.coef_)
    scores.append(reg.score(data, rts))
    
    data = data.drop(code, axis=1)

coefs = pd.DataFrame(coefs)
coefs.columns = names

coefs['SCORE'] = scores

coefs.to_csv("coefs.csv", index=False)

#==================================================

fig, ax = plt.subplots(figsize=(10,10))
ax.plot(range(38, 0, -1), scores)
ax.axis([0, 40, 0.9, 0.95])

ax.set_xticks(np.arange(0, 40, 5))
ax.set_xticks(np.arange(0, 40, 1), minor=True)
ax.xaxis.grid(True, which='major', linewidth=2)
ax.xaxis.grid(True, which='minor', linewidth=1, linestyle='--')

plt.xlabel('Number of securities')
plt.ylabel('R2')
plt.show()