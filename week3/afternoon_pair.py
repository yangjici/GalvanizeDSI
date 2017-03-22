import numpy as np
import random
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from pandas.tools.plotting import scatter_matrix
import matplotlib.pyplot as plt
from statsmodels.graphics.regressionplots import influence_plot

plt.ion()

# a utility function to only show the coefficient section of summary
from IPython.core.display import HTML
def short_summary(est):
    return HTML(est.summary().tables[1].as_html())

def dummify(df,column):
    print '{} is your baseline'.format(sorted(df[column].unique())[-1])
    dummy = pd.get_dummies(df[column]).rename(columns=lambda x: column+'_'+str(x)).iloc[:,0:len(df[column].unique())-1]
    df = df.drop(column,axis=1) #Why not inplace? because if we do inplace, it will affect the df directly
    return pd.concat([df,dummy],axis=1)


df= pd.read_csv("~/Desktop/lm_afternoon/balance.csv")



""" there seem to exist positive correlation between balance and income, limit and rating"""

dummies = pd.get_dummies(df['Married']).rename(columns = lambda x: 'Married_'+str(x))

df=pd.concat([df,dummies["Married_Yes"]],axis=1)

#2

dummies = pd.get_dummies(df['Ethnicity']).rename(columns = lambda x: 'Ethnicity_'+str(x))

df_fin=pd.concat([df,dummies[["Ethnicity_Asian" , "Ethnicity_Caucasian"]]],axis=1).drop(["Ethnicity","Married"],1)

scatter_matrix(df_fin,figsize=(10,10))
#3
est = smf.ols(formula = 'Balance ~ Unnamed: 0+ Income+ Limit+ Rating+ Cards+ Age+ Education+ Gender+ Student+ Married+ Married_Yes+ Ethnicity_Asian+ Ethnicity_Caucasian', data=df_fin).fit()
print est.summary()

#http://statsmodels.sourceforge.net/devel/examples/notebooks/generated/example_regression_plots.html

# fig = plt.figure(figsize=(12,8))
# fig = sm.graphics.plot_regress_exog(est, "Income", fig=fig)

student_resid=est.outlier_test()["student_resid"]

plt.plot(est.fittedvalues,student_resid)

#5
est1 = smf.ols(formula = 'Balance ~ Income+ Limit+ Rating+ Student', data=df_fin).fit()
student_resid=est1.outlier_test()["student_resid"]
plt.scatter(est1.fittedvalues,student_resid)
df_fin["Balance"].hist(bins=100)
'''
We observe a very heavy concentration of instances of 0
'''
for col in df_fin.columns:
    df_fin.plot(kind='scatter', y='Balance', x=col, edgecolor='none', figsize=(12, 5))


df_zeros=df_fin[df_fin["Balance"]==0]

'''
limit >3000
Rating >250
'''
df_fin_lim = df_fin[df_fin['Limit']>3000]
df_fin_rating = df_fin_lim [df_fin_lim['Rating']>250]

est2 = smf.ols(formula = 'Balance ~ Income+ Limit+ Rating+ Student', data=df_fin_rating).fit()

student_resid=est2.outlier_test()["student_resid"]
plt.scatter(est2.fittedvalues,student_resid)

df_fin_rating['Balance'].hist()
