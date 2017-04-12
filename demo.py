import pandas as pd
import quandl
import math
import datetime, time
import numpy as np
from sklearn import preprocessing, cross_validation, svm
from sklearn.linear_model import LinearRegression
import matplotlib.dates as dt
import matplotlib.pyplot as plt
from matplotlib import style

style.use('ggplot')

quandl.ApiConfig.api_key = 'xxx'
df = quandl.get_table('WIKI/PRICES', ticker='VG', date = { 'lte': '2010-04-01' })
df.set_index(['date'], inplace = True, drop=True)

df = df[['adj_open', 'adj_high', 'adj_low', 'adj_close', 'adj_volume']]
df['hl_pct']  = (df['adj_high'] - df['adj_close']) / df['adj_close'] * 100.0
df['pct_change']  = (df['adj_close'] - df['adj_open']) / df['adj_close'] * 100.0

forcast_col = 'adj_close'
df.fillna(-9999, inplace=True)

forecast_out = int(math.ceil(0.06*len(df)))
df['label'] = df[forcast_col].shift(-forecast_out)

X = np.array(df.drop(['label'],1))
X = preprocessing.scale(X)
X_lately = X[-forecast_out:]
X = X[:-forecast_out:]
df.dropna(inplace=True)

y = np.array(df['label']) # lables


X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y)

clf = LinearRegression()
clf.fit(X_train, y_train)
accuracy = clf.score(X_test, y_test)

forecast_set = clf.predict(X_lately)

print(forecast_set, accuracy, forecast_out)

df['Forecast'] = np.nan

last_date = df.iloc[-1].name
print(last_date)
last_unix = last_date.value / 1000000000
one_day = 86400
next_unix = last_unix + one_day

for i in forecast_set:
    next_date = datetime.datetime.fromtimestamp(next_unix)
    next_unix += 86400
    df.loc[next_date] = [np.nan for _ in range(len(df.columns)-1)]+[i]

df['adj_close'].plot()
df['Forecast'].plot()

plt.legend(loc=1)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()

