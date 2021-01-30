"""
The current code given is for the Assignment 2.
> Classification
> Regression
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from metrics import *

from ensemble.bagging import BaggingClassifier
from tree.base import DecisionTree
# Or use sklearn decision tree
# from linearRegression.linearRegression import LinearRegression
from sklearn.tree import DecisionTreeClassifier
########### BaggingClassifier ###################

N = 30
P = 2
NUM_OP_CLASSES = 2
n_estimators = 3
X = pd.DataFrame(np.abs(np.random.randn(N, P)))
y = pd.Series(np.random.randint(NUM_OP_CLASSES, size = N), dtype="category")

criteria = 'information_gain'
tree = DecisionTreeClassifier()
Classifier_B = BaggingClassifier(base_estimator=tree, n_estimators=n_estimators )
Classifier_B.fit(X, y)
y_hat = Classifier_B.predict(X)
Classifier_B.plot(X)
print('Criteria :', criteria)
print('Accuracy: ', accuracy(y_hat, y))
for cls in np.unique(y):
    print('Precision of {} is: '.format(cls), precision(y_hat, y, cls))
    print('Recall of {} is: '.format(cls), recall(y_hat, y, cls))
