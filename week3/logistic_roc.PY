from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np

def roc_curve(probabilities, labels):
    '''
    INPUT: numpy array, numpy array
    OUTPUT: list, list, list

    Take a numpy array of the predicted probabilities and a numpy array of the
    true labels.
    Return the True Positive Rates, False Positive Rates and Thresholds for the
    ROC curve.
    '''
    thresholds=sorted(probabilities, reverse =True)
    labels=np.array(labels)
    tpr=[]
    fpr=[]
    thres=[]
    for prob in thresholds:
        bol=np.array([1 if a > prob else 0 for a in probabilities])
        tru_pos = float(np.sum(((bol==1) & (labels==1))))
        tru_neg = float(np.sum(((bol==0) & (labels==0))))
        fal_pos = float(np.sum(((bol==1) & (labels==0))))
        fal_neg = float(np.sum(((bol==0) & (labels==1))))
        tp = tru_pos/(tru_pos+fal_neg)
        tpr.append(tp)
        fp = fal_pos/(tru_neg+fal_pos)
        fpr.append(fp)
    return tpr,fpr,thres


X, y = make_classification(n_features=2, n_redundant=0, n_informative=2,
                           n_clusters_per_class=2, n_samples=1000)
X_train, X_test, y_train, y_test = train_test_split(X, y)

model = LogisticRegression()
model.fit(X_train, y_train)
probabilities = model.predict_proba(X_test)[:, 1]



tpr, fpr, thresholds = roc_curve(probabilities, y_test)

plt.plot(fpr, tpr)
plt.xlabel("False Positive Rate (1 - Specificity)")
plt.ylabel("True Positive Rate (Sensitivity, Recall)")
plt.title("ROC plot of fake data")
plt.show()
