import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from metrics import *

from tree.randomForest import RandomForestClassifier
from tree.randomForest import RandomForestRegressor

###Write code here
da=pd.read_csv('iris.csv')
col1=da["sepal_width"]
col2=da["petal_width"]
iris=pd.merge(col1,col2,left_index=True,right_index=True)
temp=list(da["species"])
temp2=[0]*da.shape[0]
for i in range(da.shape[0]):
    if(temp[i]=="virginica"):
        temp2[i]=1
    elif(temp[i]=="setosa"):
        temp2[i]=2
    else:
        temp2[i]=3     
iris["result"]=temp2
iris=iris.sample(frac=1).reset_index(drop=True)
split_at=int(0.6*(iris.shape[0]))
X_train=iris.iloc[:split_at,:-1]
y_train=iris.iloc[:split_at,-1]
X_test=iris.iloc[split_at:,:-1]
y_test=iris.iloc[split_at:,-1]
Classifier_AB1 = RandomForestClassifier(n_estimators=10)
Classifier_AB1.fit(X_train, y_train)
y_hat = Classifier_AB1.predict(X_test)
print("Accuracy: ",accuracy(y_hat,y_test))
