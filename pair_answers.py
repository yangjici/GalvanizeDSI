import pandas as pd
from z_test import z_test
import matplotlib.pyplot as plt

plt.ion()

data = pd.read_csv("data/experiment.csv")

data.groupby([data.ab, data.landing_page]).count()

# clean = data[~((data.ab == 'treatment') & (data.landing_page == 'old_page'))]

data.groupby([data.ab, data.landing_page]).count()

# clean[clean.duplicated("user_id")==True]

clean = data.drop_duplicates("user_id", keep=False)
clean.groupby([clean.ab, clean.landing_page]).count()

# clean[clean.user_id==4472259646]
#
# clean=clean[clean.index!=147511]

#count of convertion

counts=clean.groupby([clean.landing_page,clean.converted]).count()["user_id"]

#rate of convertion

nobs_new = counts[0]+counts[1]
prop_new=float(counts[1])/nobs_new

nobs_old = counts[2]+counts[3]
prop_old=float(counts[3])/nobs_old

clean["date"] =pd.to_datetime(clean.ts,unit='s')
clean["h"] = clean.date.dt.hour

# def do_z_test(dat):
#     nobs_new=0
#     nobs_new=0
#     counts=dat.groupby([dat.landing_page,dat.converted]).count()["user_id"]
#     nobs_new = counts[0]+counts[1]
#     prop_new=float(counts[1])/nobs_new
#     nobs_old = counts[2]+counts[3]
#     prop_old=float(counts[3])/nobs_old
#     z_score, p_value, h0 = z_test.z_test(prop_old, prop_new, nobs_old, nobs_new, effect_size=0.001, two_tailed=False)
#     return p_value
#
#
# clean.groupby(clean.h).aggregate(do_z_test)

new_page = clean[clean.landing_page=='new_page']
old_page = clean[clean.landing_page=='old_page']

old_visited = old_page.groupby('h').count().cumsum()["converted"]
old_converted = old_page.groupby('h').sum()["converted"].cumsum()
old_prop = old_converted/old_visited

new_visited = new_page.groupby('h').count().cumsum()["converted"]
new_converted = new_page.groupby('h').sum()["converted"].cumsum()
new_prop = new_converted/new_visited

z_data = zip(old_prop, new_prop, old_visited, new_visited)

p_values=[z_test.z_test(*x, effect_size=0.001, two_tailed=False)[1] for x in z_data]

plt.plot(p_values)
plt.show()

# for a,b,c,d in z_data:
#
#     z_test.z_test(a,b,c,d, effect_size=0.001, two_tailed=False)
#
# z_score, p_value, h0 = z_test.z_test(prop_old, prop_new, nobs_old, nobs_new, effect_size=0.001, two_tailed=False)

"""a p value of 0.755 tell us that given that there isnt a 0.1 percent change of increase in click through rate,
there is 75 chance of observing the current statistics or more extreme, therefore we fail to reject the null
and conclude that there isnt a 0.1 percent increase in click through rate'''


#6

"""
