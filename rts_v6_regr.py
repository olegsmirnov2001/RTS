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

codes = dict(usdrub='USDRUB',
             sandp ='MINISANDP500',
             brent ='ICE.BRN',
             gold  ='comex.GC',
             eurusd='EURUSD')

dates = np.genfromtxt("data\\RTS_1-5.csv",
                      delimiter = ';',
                      dtype=str,
                      skip_header=1,
                      usecols=3)

rts = pull_data("data\\RTS_1-5.csv",
                     skip_header=1,
                     data_cols=4)

sz = rts.size

data = pull_data("data\\securities_moex.csv",
                     skip_header=2,
                     data_cols=range(78, 116),
                     dates_col=77,
                     dates_align=dates)

for code in codes:
    data_code = pull_intraday_data("data\\" + codes[code] + ".csv", dates)
    data = np.append(data, data_code.reshape((sz, 1)), axis = 1)

#==================================================

sec_names = ['GAZP','SBER','LKOH','GMKN','YNDX','NVTK','ROSN','TATN','SNGS','PLZL','MGNT','MTSS','POLY','FIVE','SNGSP','IRAO','MOEX','ALRS','NLMK','TCSG','CHMF','SBERP','VTBR','PHOR','TATNP','TRNFP','MAGN','RTKM','RUAL','HYDR','AFKS','PIKK','FEES','DSKY','AFLT','CBOM','LSRG','UPRO']
sec_weights = np.asarray([0.1468,0.1377,0.1329,0.0745,0.0459,0.0459,0.0418,0.038,0.0298,0.0211,0.0203,0.0256,0.0193,0.0205,0.016,0.0159,0.0126,0.0159,0.0127,0.0139,0.0121,0.0123,0.0135,0.0067,0.0079,0.0067,0.0064,0.0064,0.0068,0.0046,0.0045,0.0041,0.0042,0.004,0.0038,0.003,0.0032,0.0027])
# Very very bad, better read "data\\imoex_weights.csv"
other_names = list(codes.keys())
other_weights = range(100, 100 - len(other_names), -1)
names = np.concatenate((sec_names, other_names))
weights = np.concatenate((sec_weights, other_weights))

data = pd.DataFrame(data)
data = data[(-weights).argsort()]

sec_names_sorted = np.asarray(sec_names)[sec_weights.argsort()]

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

#==================================================

# rts_predicted = np.array([np.dot(reg.coef_, data[i]) for i in range(sz)])

# fig, ax = plt.subplots(figsize=(10,10))
# ax.plot(100 * rts, 100 * rts_predicted, '.')
# ax.axis([-5, 5, -5, 5])

# ax.set_xticks(np.zeros(1))
# ax.set_xticks(np.arange(-5, 5, 1), minor = True)
# ax.set_yticks(np.zeros(1))
# ax.set_yticks(np.arange(-5, 5, 1), minor = True)
# ax.grid(True, which='major', linewidth=2)
# ax.grid(True, which='minor', linewidth=1, linestyle='--')

# plt.xlabel('Real, %')
# plt.ylabel('Predicted, %')
# plt.show()

#==================================================

# Score:
# Reg v1: 0.9360584419285272
# Reg v2: 0.9414446717479138
# Reg v5: 0.9445751093173523

# Reg v1:      0.960566300365396
# Reg v1_cor:  0.9547932606361826
# Reg v2:      0.9702807180130472 with usdrub
# Reg v3:      0.9708784805702213 with s&p
# Reg v4:      0.9718925338433356 with brent and gold
# Reg v5:      0.9718925400049906 with eurusd