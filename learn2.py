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

for i in range (100):
    product=data.loc[1:,i]
    train=product[0:118]
    test=product[118:]
    # print("product"+str(i))
    # applying the SARIMA model

    prod_id = data.iloc[0:1, ]
    product_id = np.array(prod_id)
    p_id = product_id[0]
    mod = sm.tsa.statespace.SARIMAX(train,
                                order=(1, 1, 3),
                                seasonal_order=(2, 1, 1, 7),
                                enforce_stationarity=False,
                                enforce_invertibility=False)
    results = mod.fit()
    print(results.summary().tables[1])

    results.plot_diagnostics(figsize=(16, 8))
    #plt.show()
    pred = results.get_prediction(20)
    pred_ci = pred.conf_int()

    #predict the model to 28 days further
    pred_uc = results.get_forecast(steps=(29))
    pred_ci = pred_uc.conf_int()
    # ax = train.plot(label='observed', figsize=(14, 7))

    print(pred_uc.predicted_mean)
    from sklearn.metrics import mean_squared_error
    from math import sqrt
    test_arr=np.array(test)
    pred_arr=np.array(pred_uc.predicted_mean)
    qwerty = round(pred_uc.predicted_mean)
    out = np.array(qwerty)
    f = open("output.txt", "a+")
    f.write("\n")

    f.write(str(p_id[i])+" ")
    for k in range(29):
        output = out[k]
        f.write(str(round(abs(output))) + " ")
    f.close()