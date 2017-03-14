import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

with open('/Users/Gschoolstudent/Desktop/bay_area_bikeshare/201402_weather_data_v2.csv') as f:
    labels = f.readline().strip().split(',')
lbls=[(i, label) for i, label in enumerate(labels)]

cols = [2, 5, 8, 11, 14, 17]
filepath = '/Users/Gschoolstudent/Desktop/bay_area_bikeshare/201402_weather_data_v2.csv'
weather = np.loadtxt(filepath, delimiter=',', skiprows=1, usecols=cols)


fig, ax_list = plt.subplots(3, 2)
# for i in range(weather.shape[1]):
#     data = weather[:,i],label=lbls[i]
lab= map(lambda x: lbls[x][1],cols )

for subp, col in zip(ax_list.flatten(), weather.T):
    subp.plot(col)

# plt.show()

df_weather = pd.read_csv(filepath,parse_dates=['date'], index_col='date')

#matrix scatter plot

pd.scatter_matrix(df_weather[['max_temperature_f','max_humidity','max_wind_speed_m_p_h']],alpha=0.2,figsize=(3,3),diagonal = 'kde')

#plot relationship over time

df_weather[df_weather["zip"]==95113]["max_temperature_f"].plot()
