import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats.kde import gaussian_kde
import numpy as np

### Part 1: Exploratory Data Analysis (EDA)

#1  Load the file data/201402_trip_data.csv into a dataframe.
df = pd.read_csv("/Users/gschoolstudent/Downloads/data2/201402_trip_data.csv",parse_dates=['start_date', 'end_date'])

# Make 4 extra columns from the start_date column (We will use these in later questions):
df["hour"]=df["start_date"].apply(lambda x: x.hour)
df["month"]=df["start_date"].apply(lambda x: x.month)
df["dayofweek"]=df["start_date"].apply(lambda x: x.dayofweek)
df['date'] = df['start_date'].dt.date

#2 Group the bike rides by month and count the number of users per month. Plot the number of users for each month. What do you observe? Provide a likely explanation to your observation.
sav=df.groupby(df.month).count()
sav["trip_id"]

''' by plotting the month in numerical order fails to provide meaning information because date in august started in the 29th, in order to gain insight of the actual pattern we must reorder the plot based on actual time series order'''


#3 Plot the daily user count from September to December.
#Mark the mean and mean +/- 1.5 * Standard Deviation as horizontal lines on the plot. This would help you identify the outliers in your data. Describe your observations.

df2 = df[(df['month'] > 8) & (df['month'] <= 12)].groupby(['date','month']).count().trip_id
ubm = df2.reset_index()

for month in ubm.month.unique():
    plt.plot( ubm.loc[ubm.month==month].date, ubm.loc[ubm.month==month].trip_id)


mu = ubm.trip_id.mean()
st = ubm.trip_id.std()

upper = ubm.trip_id.mean() + (1.5*ubm.trip_id.std())
lower  = ubm.trip_id.mean() - (1.5*ubm.trip_id.std())

plt.axhline(y = mu)
plt.axhline(y = upper, linestyle = '--')
plt.axhline(y = lower, linestyle = '--')
plt.show()


#4  Plot the distribution of the daily user counts for all months as a histogram. Fit a KDE to the histogram.
plt.hist(df.groupby(['date']).count().trip_id,normed=True)

by_date=df.groupby(['date']).count().trip_id

my_pdf= gaussian_kde(by_date)
x=np.linspace(0,1300,len(by_date))
 #plt.plot(x,my_pdf(x),'r')

What is the distribution and explain why the distribution might be shaped as such.
""" the plot is distributed bimodally with higher density in the high user side,
the reason is that there are either a lot of users during the week day and few users during the
weekend, and the number of weekday is more by default"""


# Replot the distribution of daily user counts after binning them into weekday or weekend rides. Refit
df["weekday"]=df["dayofweek"] <=5

by_date_wknd = df[df['weekday']==False].groupby(['date']).count().trip_id
plt.hist(by_date_wknd,normed=True)
my_pdf_wknd = gaussian_kde(by_date_wknd)
plt.plot(x,my_pdf_wknd(x),'b')


by_date_wkday = df[df['weekday']==True].groupby(['date']).count().trip_id
plt.hist(by_date_wkday,normed=True)
my_pdf_wkday = gaussian_kde(by_date_wkday)
plt.plot(x,my_pdf_wkday(x),'r')


#5 Now we are going to explore hourly trends of user activity. Group the bike rides by date and hour and count the number of rides in the given hour on the given date. Make a boxplot of the hours in the day (x) against the number of users (y) in that given hour.

df.groupby(['date','hour']).count().trip_id


df5 = df.groupby(['date','hour'],as_index=True).count().trip_id

hour_flat = df5.unstack(level=1)


hour_flat.boxplot()




#6

fig, ax_list = plt.subplots(1, 2)


df_weekday = df[df["weekday"]].groupby(['date','hour'],as_index=True).count().trip_id

hour_week_flat = df_weekday.unstack(level=1)

hour_week_flat.boxplot(ax=ax_list[0])

ax_list[0].set_title('weekday')


df_weekend = df[df["weekday"]==False].groupby(['date','hour'],as_index=True).count().trip_id

hour_weekend_flat = df_weekend.unstack(level=1)

hour_weekend_flat.boxplot(ax=ax_list[1])

ax_list[1].set_title("weekend")


#7

def set_axis_options(ax, title):
   ax.set_ylim(0, 200)
   ax.set_xlabel('Hour of the Day', fontsize=14)
   ax.set_ylabel('User Freq.', fontsize=14)
   ax.title(title)

def plot_trends2(df, customer_type):
   df = df[df['subscription_type'] == customer_type] # Customer

   wkday_df = df[df['dayofweek'] <= 5]
   wkend_df = df[df['dayofweek'] > 5]

   wkday_date_hour_cnt = wkday_df.groupby(['date', 'hour']).count()['count'].reset_index()
   wkend_date_hour_cnt = wkend_df.groupby(['date', 'hour']).count()['count'].reset_index()

   wkday_gpby = wkday_date_hour_cnt.groupby('hour')
   wkend_gpby = wkend_date_hour_cnt.groupby('hour')

   wkday_lst = [wkday_gpby.get_group(hour)['count'] for hour in wkday_gpby.groups]
   wkend_lst = [wkend_gpby.get_group(hour)['count'] for hour in wkend_gpby.groups]

   fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
   ax1.boxplot(wkday_lst)
   ax2.boxplot(wkend_lst)
   set_axis_options(ax1, 'Weekday')
   set_axis_options(ax1, 'Weekend')
   plt.suptitle(customer_type, fontsize=16, fontweight='bold')
   plt.tight_layout()

plot_trends2(df, 'Subscriber')
plot_trends2(df, 'Customer')
plt.show()

