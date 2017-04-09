import sklearn
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
import numpy as np
from sklearn.datasets import load_diabetes
import matplotlib.pyplot as plt
import seaborn as sns # just 4 fun

diabetes = load_diabetes()
X = diabetes.data[:150]
y = diabetes.target[:150]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

#try Ridge reg. with alpha .5
a = 0.5
X_train_scaled = sklearn.preprocessing.scale(X_train)
fit = sklearn.linear_model.Ridge(alpha = a, normalize= True).fit(X_train_scaled,y_train)
X_test_scaled = sklearn.preprocessing.scale(X_test)

k = X.shape[1]
alphas = np.logspace(-2,2)
params = np.zeros((len(alphas),k))

train_errors=[]
test_errors=[]

for i, a in enumerate(alphas):
    fit = sklearn.linear_model.Ridge(alpha = a, normalize = True).fit(X_train_scaled,y_train)
    params[i] = fit.coef_
    train_mse=sum((fit.predict(X_train_scaled)-y_train)**2.0)/len(X_train_scaled)
    train_errors.append(train_mse)
    test_mse=sum((fit.predict(X_test_scaled)-y_test)**2.0)/len(y_test)
    test_errors.append(test_mse)

# fig = plt.figure(figsize=(14,6))
# for param in params.T:
#     plt.plot(alphas,param)

#3
plt.plot(alphas,train_errors,label="Ridge training errors")
plt.plot(alphas,test_errors,label="Ridge testing errors")
plt.legend()
plt.xlim(0,1)
# plt.show()

# we would an alpha of .7

''' PART TWO - LASSO '''

train_errors=[]
test_errors=[]

for i, a in enumerate(alphas):
    fit = sklearn.linear_model.Lasso(alpha = a, normalize = True).fit(X_train_scaled,y_train)
    params[i] = fit.coef_
    train_mse=sum((fit.predict(X_train_scaled)-y_train)**2.0)/len(X_train_scaled)
    train_errors.append(train_mse)
    test_mse=sum((fit.predict(X_test_scaled)-y_test)**2.0)/len(y_test)
    test_errors.append(test_mse)

# fig = plt.figure(figsize=(14,6))
# for param in params.T:
#     plt.plot(alphas,param)
# plt.xlim(0,10)
# plt.show()

#3
plt.plot(alphas,train_errors,label="Lasso training errors")
plt.plot(alphas,test_errors,label="Lasso testing errors")
plt.legend()
plt.xlim(0,1)
plt.show()
# ideal alpha is ~0.55
'''PART 3 - MODEL SELECTION

When comparing Ridge vs. Lasso on the same plot, we observe that the testing error for Ridge is lower across the board (including at the optimal alpha values).
'''


diabetes = load_diabetes()
X = diabetes.data[:150]
y = diabetes.target[:150]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

#try Ridge reg. with alpha .5
a = 0.5
X_train_scaled = sklearn.preprocessing.scale(X_train)
fit = sklearn.linear_model.Ridge(alpha = a, normalize= True).fit(X_train_scaled,y_train)
X_test_scaled = sklearn.preprocessing.scale(X_test)

k = X.shape[1]
alphas = np.logspace(-2,2)
params = np.zeros((len(alphas),k))

train_errors=[]
test_errors=[]

for i, a in enumerate(alphas):
    fit = sklearn.linear_model.Ridge(alpha = a, normalize = True).fit(X_train_scaled,y_train)
    params[i] = fit.coef_
    train_mse=sum((fit.predict(X_train_scaled)-y_train)**2.0)/len(X_train_scaled)
    train_errors.append(train_mse)
    test_mse=sum((fit.predict(X_test_scaled)-y_test)**2.0)/len(y_test)
    test_errors.append(test_mse)



# fig = plt.figure(figsize=(14,6))
# for param in params.T:
#     plt.plot(alphas,param)

#3
plt.plot(alphas,train_errors,label="Ridge training errors")
plt.plot(alphas,test_errors,label="Ridge testing errors")
plt.legend()
plt.xlim(0,1)
# plt.show()

# we would an alpha of .7

''' PART TWO - LASSO '''

train_errors=[]
test_errors=[]

for i, a in enumerate(alphas):
    fit = sklearn.linear_model.Lasso(alpha = a, normalize = True).fit(X_train_scaled,y_train)
    params[i] = fit.coef_
    train_mse=sum((fit.predict(X_train_scaled)-y_train)**2.0)/len(X_train_scaled)
    train_errors.append(train_mse)
    test_mse=sum((fit.predict(X_test_scaled)-y_test)**2.0)/len(y_test)
    test_errors.append(test_mse)

# fig = plt.figure(figsize=(14,6))
# for param in params.T:
#     plt.plot(alphas,param)
# plt.xlim(0,10)
# plt.show()

#3
plt.plot(alphas,train_errors,label="Lasso training errors")
plt.plot(alphas,test_errors,label="Lasso testing errors")
plt.legend()
plt.xlim(0,1)
plt.show()
# ideal alpha is ~0.55
'''PART 3 - MODEL SELECTION

When comparing Ridge vs. Lasso on the same plot, we observe that the testing error for Ridge is lower across the board (including at the optimal alpha values).

With all of the data included, we note that testing AND training error is strictly increasing with alpha, meaning that regularization does not correct any overfitting and is therefore unnecessary.
'''
