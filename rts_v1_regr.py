import numpy as np
# import pandas as pd
from sklearn.linear_model import LinearRegression
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

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
                data_aligned[i] = data_aligned[i-1]
            #else:
                #pass #Bad practise, better fill with filling_values
        
        return data_aligned
    
    return data

def pull_intraday_data(fname,
                       dates_align,
                       delimiter = ';',
                       skip_header = 1,
                       missing_values = '',
                       filling_values = 0,
                       open_col = 1,
                       close_col = 2,
                       dates_col = 0):
    data_open = pull_data(fname,
                          delimiter,
                          skip_header,
                          missing_values,
                          filling_values,
                          open_col,
                          dates_col,
                          dates_align)
    data_close = pull_data(fname,
                           delimiter,
                           skip_header,
                           missing_values,
                           filling_values,
                           close_col,
                           dates_col,
                           np.concatenate([[(datetime.strptime(dates_align[0],
                                    "%Y-%m-%d") - timedelta(days=1)).strftime("%Y-%m-%d")],
                                            dates_align]))
                                
    data = np.empty(dates_align.size, float)
    for i in range(len(dates_align)):
        data[i] = data_open[i] / data_close[i] - 1
    return data

#==================================================

dates = np.genfromtxt("data\\RTS_1-5.csv",
                      delimiter = ';',
                      dtype=str,
                      skip_header=1,
                      usecols=3)

data_rts = pull_data("data\\RTS_1-5.csv",
                     skip_header=1,
                     data_cols=4)

sz = data_rts.size

data_usdrub = pull_intraday_data("data\\USDRUB.csv", dates)

data_sec = pull_data("data\\securities_moex.csv",
                     skip_header=2,
                     data_cols=range(78, 116),
                     dates_col=77,
                     dates_align=dates)

data_weights = pull_data("data\\imoex_weights.csv",
                         skip_header=1,
                         data_cols=range(1, 39),
                         dates_col=0,
                         dates_align=dates,
                         use_yesterday=True)

data_rts_rub = (1 + data_rts) * (1 + data_usdrub) - 1

data_predicted_weights = np.empty(data_rts.size, float)
for i in range(data_rts.size):
    data_predicted_weights[i] = np.dot(data_sec[i], data_weights[i])

reg = LinearRegression().fit(data_sec, data_rts_rub)
print (reg.coef_)
# print (reg.score(data_sec, data_rts))

data_predicted_reg = np.array([np.dot(reg.coef_, data_sec[i]) for i in range(sz)])

data_predicted_reg_usd = (1 + data_predicted_reg) / (1 + data_usdrub) - 1

#==================================================

fig, ax = plt.subplots(figsize=(10,10))
ax.plot(100 * data_rts, 100 * data_predicted_reg_usd, '.')
ax.axis([-5, 5, -5, 5])

ax.set_xticks(np.zeros(1))
ax.set_xticks(np.arange(-5, 5, 1), minor = True)
ax.set_yticks(np.zeros(1))
ax.set_yticks(np.arange(-5, 5, 1), minor = True)
ax.grid(True, which='major', linewidth=2)
ax.grid(True, which='minor', linewidth=1, linestyle='--')

plt.xlabel('Real, %')
plt.ylabel('Predicted, %')
plt.show()

#==================================================

print(np.corrcoef(data_rts, data_predicted_reg_usd)[0][1])
#print(np.corrcoef(data_fut, data_predicted)[0][1])

# Reg 1:      0.960566300365396