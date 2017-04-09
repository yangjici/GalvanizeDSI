import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from statsmodels.discrete.discrete_model import Logit
from statsmodels.tools.tools import add_constant
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from itertools import chain

grad= pd.read_csv("/Users/datascientist/Desktop/logistic_regression/grad.csv")

grad.describe()

grad.info()

dat=pd.crosstab(grad["admit"],grad["rank"])

colsum=np.sum(dat.values,axis=0)

prop=dat.apply(lambda x: x/colsum,axis=1 )

prop.T.plot(kind="bar")

grad["gpa"].hist()

grad["gre"].hist()

grad["admit"].hist()

np.mean(grad['admit'])

plt.show()

# Part 3
y = grad.admit
X = grad.iloc[:, 1:]
X = add_constant(X)
logit = Logit(y, X).fit()
logit.summary()

X_train, X_test, y_train, y_test = train_test_split(X, y)

def k_fold_logistic(X_train,y_train):
    err_rate, index, num_folds = 0, 0, 5
    m = len(X_train)
    kf = KFold(n_splits=num_folds,shuffle=True,random_state=6)
    logit = LogisticRegression()
    accuracy=[]
    recall=[]
    precision=[]

    for train, test in kf.split(X_train):
        logit.fit(X_train.iloc[train], y_train.iloc[train])
        probs = logit.predict(X_train.iloc[test])
        predicted_label = np.array([1 if prob >= 0.5 else 0 for prob in probs])
        true_labels = np.array(y_train.iloc[test])
        FN = np.sum((true_labels == 1) & (predicted_label == 0))
        TP = np.sum((true_labels == 1) & (predicted_label == 1))
        TN = np.sum((true_labels == 0) & (predicted_label == 0))
        FP = np.sum((true_labels == 0) & (predicted_label == 1))
        acc=(TP+TN)/float(len(true_labels))
        pre=TP/float((TP+FP))
        re=TP/float((TP+FN))
        accuracy.append(acc)
        recall.append(re)
        precision.append(pre)
    return accuracy,recall,precision

res=k_fold_logistic(X_train,y_train)

res_mean=[np.mean(x) for x in res]
dummy=pd.get_dummies(grad["rank"]).loc[:,:3]
#4

df_rank=pd.concat ([grad, dummy ],axis=1).drop("rank",axis=1)

#5

y_rank = df_rank.admit
X_rank = df_rank.iloc[:, 1:]
X_rank=add_constant(X_rank)

X_train_rank, X_test, y_train_rank, y_test = train_test_split(X_rank, y_rank)


res_rank=k_fold_logistic(X_train_rank,y_train_rank)
res_mean_rank=[np.mean(x) for x in res_rank]


# 6
def roc_curve(probabilities, labels):
    '''
    INPUT: numpy array, numpy array
    OUTPUT: list, list, list
    Take a numpy array of the predicted probabilities and a numpy array of the
    true labels.
    Return the True Positive Rates, False Positive Rates and Thresholds for the
    ROC curve.
    '''
    TPR = []
    FPR = []

    for threshold in probabilities:
        predicted_labels = np.array([1 if el >= threshold else 0 for el in probabilities])
        FN = np.sum((labels == 1) & (predicted_labels == 0))
        TP = np.sum((labels == 1) & (predicted_labels == 1))
        TN = np.sum((labels == 0) & (predicted_labels == 0))
        FP = np.sum((labels == 0) & (predicted_labels == 1))
        TPR.append(float(TP) / (TP + FN))
        FPR.append(float(FP) / (TN + FP))

    idx = np.argsort(probabilities)

    np.argsort(probs)

    return np.array(TPR)[idx], np.array(FPR)[idx], probabilities[idx]


logit = LogisticRegression()
logit.fit(X_train_rank, y_train_rank)


probs = logit.predict_proba(X_test)[:, 1:]
probs=np.array([x[0] for x in probs])

tpr, fpr, thresholds = roc_curve(probs, np.array(y_test))

plt.plot(fpr, tpr)
plt.plot([0, 1], [0, 1], ls="--", c=".3", alpha=0.5)
plt.xlabel("False Positive Rate (1 - Specificity)")
plt.ylabel("True Positive Rate (Sensitivity, Recall)")
plt.axvline(0.4)
plt.axhline(0.6)
plt.title("ROC plot of fake data")
plt.show()

logit = LogisticRegression()
logit.fit(X_rank, y_rank)
X_rank.head()
logit.coef_

#5


design=np.hstack((np.array(list(np.mean(X_rank,axis=0)[:3])*4).reshape(4,3),np.vstack((np.identity(3), np.array([0.,0.,0.]).reshape(1,3)))))


res=logit.predict_proba(design)[:,1:]

plt.plot(res)

plt.show()

plt.plot([np.log(a/(1-a)) for a in res])
