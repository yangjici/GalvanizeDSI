import statsmodels as sm
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.statespace.sarimax import SARIMAX

sm.__version__

df = pd.read_json('logins.json')

df=df.reset_index()

df.index =df[0]

df['count']=[1 for x in df.index]

df=df.drop(['index',0],axis=1)

hr_count = df.resample('1D',label = 'count')

series = pd.Series(df['count'],index=pd.DatetimeIndex(df.index))

by_day=series.resample('1D').sum()

by_hr= series.resample('1H').sum()


#plot
#
# plt.plot(by_day.index,by_day)
#
# plt.plot(by_hour.index,by_hour)

dayofweek = pd.DataFrame(by_day)

dayofweek['dayofweek']=dayofweek.index.dayofweek

dayofweek['weekend'] = dayofweek['dayofweek'].isin([5,6])


#plot to show weekend

plt.plot(by_day.index,by_day)

plt.fill_between( by_day.index,by_day,where =dayofweek['weekend'])


def plot_acf_pacf(your_data, lags):
   fig = plt.figure(figsize=(12,8))
   ax1 = fig.add_subplot(211)
   fig = plot_acf(your_data, lags=lags, ax=ax1)
   ax2 = fig.add_subplot(212)
   fig = plot_pacf(your_data, lags=lags, ax=ax2)
   plt.show()

plot_acf_pacf(by_day,lags=28)


diff1=by_day.diff(periods=1)

diff7=by_day.diff(periods=7)

plot_acf_pacf(diff1[1:],lags=28)

#detrend

def detrend(data):
    x = range(len(data))
    lm=sm.OLS(data,x)
    lm=sm.OLS(data,x).fit()
    trend = lm.predict(x)
    residual = by_day-by_day_trend
    return residual

by_day_residual = detrend(by_day)

diff7_res=by_day_residual.diff(periods=7)


plot_acf_pacf(diff7_res[7:],lags=28)

#SARIMA

model=SARIMAX(by_day_residual, order=(1,1,1), seasonal_order=(1,1,1,7),enforce_invertibility=False).fit()

plt.plot(model.resid.index,model.resid)

plot_acf_pacf(model.resid,lags=28)



# Perform on hourly data

by_hour=series.resample('1H').sum()
