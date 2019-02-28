#importing all the package

import numpy as np
import math
import itertools
import statsmodels.api as sm
import pandas as pd
import matplotlib.pylab as plt
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 15, 6
from statsmodels.tsa.arima_model import ARIMA

# read the data from the pro_dist.txt
data = pd.read_csv("product_distribution_training_set.txt", delimiter='\t',header=None)
#transpose the data for quick access to the data
data=data.T
new_header = data.iloc[0]
data.columns = new_header
data = data.drop(data.index[[0]])
# compute the sum of total number of key products sold each day
sum = np.sum(data,axis=1).astype(float)
print(sum)
# convert the sum into an array
sumarray = np.array(sum)
# convert the sumarray into the series
ts = pd.Series(sumarray)
# plot the values of product sold each day
ts.plot(color = 'yellow')
# plt.show()

# testing the stationarity of the series
from statsmodels.tsa.stattools import adfuller


def test_stationarity(timeseries):
    # Determing rolling statistics
    rolmean = timeseries.rolling(4).mean()
    #rolstd = pd.rolling_std(timeseries, window=4)
    rolstd = timeseries.rolling(4).std()
    # Plot rolling statistics:
    orig = plt.plot(timeseries, color='blue', label='Original')
    mean = plt.plot(rolmean, color='red', label='Rolling Mean')
    std = plt.plot(rolstd, color='black', label='Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show(block=False)

    # Perform Dickey-Fuller test:
    print('Results of Dickey-Fuller Test:')
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used'])
    for key, value in dftest[4].items():
        dfoutput['Critical Value (%s)' % key] = value
    print(dfoutput)

test_stationarity(sum)

from statsmodels.tsa.seasonal import seasonal_decompose
decomposition = seasonal_decompose(sum,freq=10)

trend = decomposition.trend
seasonal = decomposition.seasonal
residual = decomposition.resid

plt.subplot(411)
plt.plot(sum, label='Original')
plt.legend(loc='best')
plt.subplot(412)
plt.plot(trend, label='Trend')
plt.legend(loc='best')
plt.subplot(413)
plt.plot(seasonal,label='Seasonality')
plt.legend(loc='best')
plt.subplot(414)
plt.plot(residual, label='Residuals')
plt.legend(loc='best')
plt.tight_layout()

train=sum[0:118]
test=sum[118:]
print(train)

# applying the SARIMA model
mod = sm.tsa.statespace.SARIMAX(train,
                                order=(2, 1, 2),
                                seasonal_order=(2, 1, 1, 7),
                                enforce_stationarity=False,
                                enforce_invertibility=False)
results = mod.fit()
print(results.summary().tables[1])

results.plot_diagnostics(figsize=(16, 8))
plt.show()
pred = results.get_prediction(20)
pred_ci = pred.conf_int()
train.plot(label='observed',color='red')
ax = train.plot(label='observed')
pred.predicted_mean.plot(ax=ax, label='One-step ahead Forecast', alpha=.7, figsize=(14, 7))
ax.fill_between(pred_ci.index,
                pred_ci.iloc[:, 0],
                pred_ci.iloc[:, 1], color='k', alpha=.2)
ax.set_xlabel('Days')
ax.set_ylabel('Sales')
plt.legend()
plt.show()

#predict the model to 28 days further
pred_uc = results.get_forecast(steps=29)
pred_ci = pred_uc.conf_int()
# ax = train.plot(label='observed', figsize=(14, 7))
# pred_uc.predicted_mean.plot(ax=ax, label='Forecast')
ax.fill_between(pred_ci.index,
                pred_ci.iloc[:, 0],
                pred_ci.iloc[:, 1], color='k', alpha=.25)

print(pred_uc.predicted_mean)

qwerty =  round(pred_uc.predicted_mean)
out = np.array(qwerty)
f = open("output.txt", "a+")
f.write("0 ")

for k in range (29):
    output = out[k]
    f.write(str(round(int(output)))+ " ")

f.close()