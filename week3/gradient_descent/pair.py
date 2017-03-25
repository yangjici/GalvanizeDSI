import numpy as np
from numpy import exp
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from statsmodels.discrete.discrete_model import Logit
from statsmodels.tools.tools import add_constant
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from itertools import chain





X,y=make_classification(n_samples=100,n_features=2,n_informative=2,n_redundant=0,n_classes=2,random_state=0)



plt.scatter(X[:,:1],y,c=y)

plt.show()
'''
Part 2
1.
h(xi) = 1/(1+exp(-Beta*xi))
'''

xa1=np.array([3,0]).reshape(2,1)

b11=np.array([1,1])

ha1=1/(1+exp(-b11.dot(xa1)))

h_array=[0.731,0.982,0.9525]

xa2 = np.array([2,2]).reshape(2,1)
#ha= np.array([1/(1+np.exp(-1*a)) for a in xa])

y1=np.array([1,0,0])

def cost(h,y):
     return y*np.log(h)+(1-y)*np.log(1-h)

value_cost=-(sum([cost(h,y) for h,y in zip(h_array,y1)]))



'''value cost = 7.3778'''



#2.


''' b1'''

j1= np.array([a-b for a,b in zip(h_array,y1)])
xj1=np.array([0,2,3])

J_1=sum([a*b for a,b in zip(j1,xj1)])

xj2=np.array([1,2,0])

J_2=sum([a*b for a,b in zip(j1,xj2)])

"""
gradient is [4.8215,1.695]
"""




"""testing

""""

y=np.array([1,0,0])

X=np.array([[0,1],[2,2],[3,0]]).reshape(3,2)



coeffs=np.array([1,1]).reshape(2,1)

predict_proba(X, coeffs)

predict(X, coeffs, threas=0.5)

cost(X, y, coeffs)

gradient(X,coeffs,y)

X = np.array([[0, 1], [2, 2]])
y = np.array([1, 0])
coeffs = np.array([1, 1])
f.cost(X, y, coeffs)







################ PART 3
import os
os.chdir("/Users/datascientist/Desktop/gradient-descent-files/src")


import logistic_regression_functions as f
from GradientDescent import GradientDescent

gd = GradientDescent(f.cost, f.gradient, f.predict,fit_intercept=False)
gd.run(X, y)
print "coeffs:", gd.coeffs
predictions = gd.predict(X)

#PART 4

logit=LogisticRegression()

logit.fit(X, y)

logit.coef_

#PART 5
