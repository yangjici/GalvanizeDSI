import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, confusion_matrix
import re
from sklearn.cross_validation import train_test_split
import matplotlib.pyplot as plt
from roc import plot_roc
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

def fit_forest(X_train,y_train,X_test,y_test,num_trees=10,oob_score=False,max_features='auto'):
    forest = RandomForestClassifier(n_estimators=num_trees,oob_score = oob_score,max_features=max_features)
    forest.fit(X_train,y_train)
    prediction = forest.predict(X_test)
    accuracy = forest.score(X_test,y_test)
    precision= precision_score(y_test,prediction)
    recall= recall_score(y_test,prediction)
    confusion= confusion_matrix(y_test,prediction)

    return forest, prediction, accuracy, precision, recall, confusion

def fit_model(X_train,y_train,X_test,y_test,clf_class, **kwargs):
    model = clf_class(**kwargs)
    model.fit(X_train,y_train)
    prediction = forest.predict(X_test)
    accuracy = forest.score(X_test,y_test)
    precision= precision_score(y_test,prediction)
    recall= recall_score(y_test,prediction)
    confusion= confusion_matrix(y_test,prediction)

    return forest, accuracy, precision, recall, confusion

df = pd.read_csv('../data/churn.csv')
new_cols=map(lambda x: re.sub(" ","",x),df.columns.values)
df.columns = new_cols
df['VMailPlan'] = df['VMailPlan'].apply(lambda x: True if x=='yes' else False)
df["Int'lPlan"] = df["Int'lPlan"].apply(lambda x: True if x=='yes' else False)
delete = ["State","AccountLength","AreaCode","Phone"]
df_new=df.drop(delete,axis=1)
y=df_new["Churn?"].apply(lambda x: True if x=='True.' else False)
X=np.array(df_new.drop("Churn?",axis=1))
X_train, X_test, y_train, y_test = train_test_split(X, y)

forest, prediction, accuracy, precision, recall, confusion = fit_forest(X_train,y_train,X_test,y_test)

forest2, prediction2, accuracy2, precision2, recall2, confusion2 = fit_forest(X_train,y_train,X_test,y_test,oob_score=True)
oob = forest2.oob_score_
feature_importance = forest2.feature_importances_
column_names = df_new.drop("Churn?",axis=1).columns.values

sorted_by_second = np.array(sorted(zip(feature_importance,column_names), key=lambda tup: tup[0], reverse=True))

# plt.figure(figsize=(12,8))
# plt.bar(range(len(sorted_by_second[:,0])),list(sorted_by_second[:,0]))
#
# plt.xticks(range(len(sorted_by_second[:,0])),list(sorted_by_second[:,1]),rotation=90)
# plt.show()

#13
num_trees=[5,20,50,100,150,300]

accuracies=[]

for num in num_trees:
    res=fit_forest(X_train,y_train,X_test,y_test,num_trees=num,oob_score=False)
    accuracies.append(res[2])

# plt.plot(num_trees,accuracies,"o")
# plt.show()

num_features = np.arange(1,len(column_names))
accuracies_features=[]
for num in num_features:
    res=fit_forest(X_train,y_train,X_test,y_test,num_trees=300,oob_score=False,max_features=num)
    accuracies_features.append(res[2])

plt.plot(num_features,accuracies_features,"o")
plt.show()

model_names = [LogisticRegression,KNeighborsClassifier,DecisionTreeClassifier]

models = [fit_model(X_train,y_train,X_test,y_test,model) for model in model_names]

forest = fit_forest(X_train,y_train,X_test,y_test,num_trees=300,oob_score=False,max_features=7)

for model in model_names:
    plot_roc(X, y, model)
plot_roc(X, y, RandomForestClassifier,n_estimators=300,oob_score=False,max_features=7)
