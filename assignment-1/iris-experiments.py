from numpy.lib.function_base import percentile
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tree.base import DecisionTree
from metrics import *

np.random.seed(42)

# Read IRIS data set
# ...
# 
iris=pd.read_csv('iris.csv')
iris=iris.sample(frac=1).reset_index(drop=True)
split_at=int(0.7*(iris.shape[0]))
X_train=iris.iloc[:split_at,:-1]
y_train=iris.iloc[:split_at,-1]
X_test=iris.iloc[split_at:,:-1]
y_test=iris.iloc[split_at:,-1]
model=DecisionTree()
model.fit(X_train,y_train)
y_out=model.predict(X_test)
print("Accuracy is: ",accuracy(y_out,y_test))
for group in np.unique(y_test):
    print("Precision of {} is: {}".format(group,precision(y_out,y_test,group)))
    print("Recal of {} is: {}".format(group,recall(y_out,y_test,group)))
