'''
identifying distributions
1. poisson where each page is interval
 let x be the number of mistakes on page
p(k=0 for 1 page) ~ poisson(lambda=2, k=0)
 p(2,0)=lambda^0(e-2)/0! = 0.13533528323

2. binomial where p=0.1 for defective, n = 20, x=2
let x be the number of defective
X ~ Binomial(n=20, p=0.1)
Probability of (x=2, n=20, p=0.1) = 20c2*0.1^2*0.9^18 = 0.28517980706

3. exponential lambda = 30/hr * 1hr/60min = 0.5 /min
p(x>3)~ expon(lambda = 0.5)
p=1-p(x<=3) = 1-0.7769 = 0.2231

4. normal distribution with a cdf call
z score = 7/standard dev = 7/5=1.4
'''
from scipy.stats import norm
p_xg127=1-norm.cdf(1.4)
'''
0.080756659233771066

5. Geometric
p=0.08
X~Geomet(p=0.08)
E(Y) = (1-p)/p = .92/.08 = 11.5

6. Uniform Continuous
X ~ Uniform(0,20)
CDF= 1-(15/20) = 0.25

'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pylab import plot,show,hist
from scipy.stats.kde import gaussian_kde

df = pd.read_csv("~/Downloads/data/admissions.csv")


def my_cov(dat):
    mat= np.zeros(shape=(dat.shape[1],dat.shape[1]))
    for i in range(dat.shape[1]):
        for j in range(dat.shape[1]):
            mat[i,j]=sum(dat.ix[:,i]*dat.ix[:,j]-np.mean(dat.ix[:,i])*np.mean(dat.ix[:,j]))/(len(dat.ix[:,0])-1)
    return mat
"""
>>> my_cov(df)
array([[  3.32941046e+08,   4.01529909e+03,  -1.22632628e+03],
       [  4.01529909e+03,   8.78911925e-02,  -2.87852599e-02],
       [ -1.22632628e+03,  -2.87852599e-02,   1.12977442e+02]])
"""

def normalize(dat):
    mat= np.zeros(shape=(dat.shape[1],dat.shape[1]))
    for i in range(dat.shape[1]):
        for j in range(dat.shape[1]):
            mat[i,j]=sum(dat.ix[:,i]*dat.ix[:,j]-np.mean(dat.ix[:,i])*np.mean(dat.ix[:,j]))/(len(dat.ix[:,0])-1)/(np.std(dat.ix[:,i])*np.std(dat.ix[:,j]))

#            mat.columns = ['family_income','gpa' ,'parent_avg_age']
#            mat.index= ['family_income','gpa' ,'parent_avg_age']
    mat=pd.DataFrame(mat, index=['family_income','gpa' ,'parent_avg_age'], columns=['family_income','gpa' ,'parent_avg_age'])
    return mat

"""
>>> normalize(df)
                family_income       gpa  parent_avg_age
family_income        1.000091  0.742337       -0.006324
gpa                  0.742337  1.000091       -0.009136
parent_avg_age      -0.006324 -0.009136        1.000091

"""

def income_code(dat):
    if dat["family_income"]<26832:
        val = "low income"
    elif dat["family_income"]>26832 and dat["family_income"]<37510:
        val = "medium income"
    else:
        val = "high income"
    return val

df["income_group"] = df.apply(income_code,axis=1)



df.groupby("income_group")["gpa"].plot(kind="kde")
plt.legend(["low income", "medium income", "high income"], loc='upper left')
