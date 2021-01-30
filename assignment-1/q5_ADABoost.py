"""
The current code given is for the Assignment 2.
> Classification
> Regression
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# from sklearn.tree import DecisionTreeClassifier
from metrics import *
from ensemble.ADABoost import AdaBoostClassifier
# from tree.base import DecisionTree
# Or you could import sklearn DecisionTree
# from linearRegression.linearRegression import LinearRegression

np.random.seed(42)

########### AdaBoostClassifier on Real Input and Discrete Output ###################


# N = 30
# P = 2
# NUM_OP_CLASSES = 2
# n_estimators = 3
# X = pd.DataFrame(np.abs(np.random.randn(N, P)))
# y = pd.Series(np.random.randint(NUM_OP_CLASSES, size = N))
# y = pd.Series(np.where(y==0,-1,1))
# criteria = 'information_gain'
# Classifier_AB = AdaBoostClassifier(n_estimators=n_estimators )
# Classifier_AB.fit(X, y)
# y_hat = Classifier_AB.predict(X)
# # [fig1, fig2] = Classifier_AB.plot()
# print('Criteria :', criteria)
# print('Accuracy: ', accuracy(y_hat, y))
# for cls in np.unique(y):
#     print('Precision of {} is: '.format(cls), precision(y_hat, y, cls))
#     print('Recall of {} is: '.format(cls), recall(y_hat, y, cls))

# del(Classifier_AB)

##### AdaBoostClassifier on Iris data set using the entire data set with sepal width and petal width as the two features
da=pd.read_csv('iris.csv')
col1=da["sepal_width"]
col2=da["petal_width"]
label=np.array(da["species"])
label=np.where(label=="virginica",1,-1)
iris=pd.merge(col1,col2,left_index=True,right_index=True)
iris["Truth"]=label
iris=iris.sample(frac=1).reset_index(drop=True)
split_at=int(0.6*(iris.shape[0]))
X_train=iris.iloc[:split_at,:-1]
y_train=iris.iloc[:split_at,-1]
X_test=iris.iloc[split_at:,:-1]
y_test=iris.iloc[split_at:,-1]
Classifier_AB1 = AdaBoostClassifier(n_estimators=3 )
Classifier_AB1.fit(X_train, y_train)
y_hat = Classifier_AB1.predict(X_test)
print(list(y_hat),list(y_test))
print("Accuracy: ",accuracy(y_hat,y_test))
Classifier_AB1.plot(X_test)
