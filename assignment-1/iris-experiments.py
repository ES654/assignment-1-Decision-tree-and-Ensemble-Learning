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

#Accuracy of all five models 
fold=int(0.2*(iris.shape[0]))
for i in range(5):
    n_split1=i*fold
    n_split2=n_split1+fold
    X_test1=iris.iloc[n_split1:n_split2,:-1].reset_index(drop=True)
    y_test1=pd.Series(list(iris.iloc[n_split1:n_split2,-1]))
    X_train1=iris.iloc[:n_split1,:-1].append(iris.iloc[n_split2:,:-1]).reset_index(drop=True)
    y_train1=pd.Series(list(iris.iloc[:n_split1,-1].append(iris.iloc[n_split2:,-1])))
    model=DecisionTree()
    model.fit(X_train1,y_train1)
    y_hat=model.predict(X_test1)
    acc=accuracy(y_hat,y_test1)
    print("Accuracy of model{} is: ".format(i),acc)

# Nested cross validation to find optimal depth
opt_depth={0:dict(),1:dict(),2:dict(),3:dict(),4:dict()}
fold=int(0.2*(iris.shape[0]))
for i in range(5):
    n_split1=i*fold
    n_split2=n_split1+fold
    X_validation=iris.iloc[n_split1:n_split2,:-1].reset_index(drop=True)
    y_validation=pd.Series(list(iris.iloc[n_split1:n_split2,-1]))
    X_train1=iris.iloc[:n_split1,:-1].append(iris.iloc[n_split2:,:-1]).reset_index(drop=True)
    y_train1=pd.Series(list(iris.iloc[:n_split1,-1].append(iris.iloc[n_split2:,-1])))
    maxacc=0
    op_dep=2
    for k in range(2,8):
        model=DecisionTree(max_depth=k)
        model.fit(X_train1,y_train1)
        y_cap=model.predict(X_validation)
        acc=accuracy(y_cap,y_validation)
        if(acc>maxacc):
            op_dep=k
            maxacc=acc
    opt_depth[i]["Depth"]=op_dep
    opt_depth[i]["Accuracy"]=maxacc
print(opt_depth)
best_depth=0
best_accuracy=0
for i in range(len(opt_depth)):
    if(opt_depth[i]["Accuracy"]>best_accuracy):
        best_depth=opt_depth[i]["Depth"]
        best_accuracy=opt_depth[i]["Accuracy"]
print("Best depth is: {} and its accuracy is: {}".format(best_depth,best_accuracy))
