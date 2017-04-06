import pandas as pd
from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegression as LR
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.ensemble import GradientBoostingClassifier as GBC
from sklearn.svm import SVC

churn = pd.read_csv('churn.csv')
churn = churn.drop(['State','Area Code','Phone'], axis = 1)

churn["Int'l Plan"] = churn["Int'l Plan"].map(dict(yes=1, no=0))
churn["VMail Plan"] = churn["VMail Plan"].map(dict(yes=1, no=0))
churn["Churn?"] = churn["Churn?"] == 'True.'
churn["Churn?"] = churn["Churn?"].astype(int)

X = churn.drop('Churn?', axis = 1).values
y= churn['Churn?'].values
X_train, X_test, y_train, y_test = train_test_split(X,y)

test_probs = np.array([0.2, 0.6, 0.4])
test_labels = np.array([0, 0, 1])
test_cost_benefit = np.array([[0, -3], [0, 6]])


churn_matrix= np.array([[0, -2],[-10, -5]])

def profit_curve(cost_benefit, predicted_probs, labels):
    sorted_prob = np.append(predicted_probs, 1)
    sorted_prob = sorted(sorted_prob, reverse = True)
    results = []
    for threshold in sorted_prob:
        y_pred = [1 if prob > threshold else 0 for prob in predicted_probs]
        con_mat = confusion_matrix(labels, y_pred)
        profit = con_mat * cost_benefit
        profit = profit.sum()
        expected = profit / float(len(y_pred))
        results.append(expected)
    return results

# profits = profit_curve(test_cost_benefit, test_probs, test_labels)
#
# percentages = np.arange(0, 100, 100. / len(profits))
# plt.plot(percentages, profits, label='toy data')
# plt.title("Profit Curve")
# plt.xlabel("Percentage of test instances (decreasing by score)")
# plt.ylabel("Profit")
# plt.legend(loc='best')
#plt.show()

def ch(model, cost_benefit, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict_proba(X_test)[:,1]
    profits = profit_curve(cost_benefit, y_pred, y_test)
    percentages = np.arange(0, 100, 100. / len(profits))
    plt.plot(percentages, profits, label='model{}'.format(model.__class__.__name__))



cost_benefit = np.array([[0, -3], [0, 6]])
models = [RF(), LR(), GBC(), SVC(probability=True)]
for model in models:
    ch(model, cost_benefit, X_train, X_test, y_train, y_test)
plt.title("Profit Curves")
plt.xlabel("Percentage of test instances (decreasing by score)")
plt.ylabel("Profit")
plt.legend(loc='best')
plt.show()
