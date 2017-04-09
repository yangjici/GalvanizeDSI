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
from pylab import plot,show,hist,title
from scipy.stats.kde import gaussian_kde
import scipy.stats as scs

df = pd.read_csv("/Users/twilightidol/Downloads/data (1)/admissions.csv")


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

df_new=df

df_new["income_group"] = df_new.apply(income_code,axis=1)



df_new.groupby("income_group")["gpa"].plot(kind="kde")
plt.legend(["high income", "low income", "medium income"], loc='upper left')

show()


nintey_percentile =df_new.groupby("income_group")["gpa"].agg(lambda x: np.percentile(x,90))


'''
PART 3
'''

#1

df2 = pd.read_csv("/Users/twilightidol/Downloads/data (1)/admissions_with_study_hrs_and_sports.csv")

x=df2["hrs_studied"]

y=df2["gpa"]

slope, intercept, r_value, p_value, std_err = scs.linregress(x, y)

line = slope*x+intercept

plot(x,y,'o', x, line)

title("Linear fit of Hours studied Versus GPA")

show()


"""
3: pearson correlation is great at detecting purely first degree linear relationship
while spearman correlation is good at detecting monotonic relationship in general,
since the relationship is curvilinear, spearman is more sensitive to the correlation
than pearson.
"""

'''
PART 4
'''

def profit_range(n):
    profits=[]
    for _ in range(n):
        views=np.random.randint(5000,6000)
        conversions=scs.binom.rvs(views,0.12)
        whole_sale_prop= scs.binom.rvs(conversions,0.2)/conversions
        profits.append(conversions*(whole_sale_prop*50+(1-whole_sale_prop)*60))
    return (np.percentile(profits,2.5),np.percentile(profits,97.5))

'''
given 10000 simulations, 95 percentile of profit falls in the range of
34980 and 44341z

'''









if __name__ == '__main__':
    print my_cov(df.ix[:,:3])
    print normalize(df.ix[:,:3])
    print df.head()
    print df_new.head()
    print nintey_percentile
    print scs.pearsonr(x,y)
    print scs.spearmanr(x,y)
