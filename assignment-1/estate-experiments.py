
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tree.base import DecisionTree
from metrics import *
from sklearn.tree import DecisionTreeRegressor
np.random.seed(42)

# Read real-estate data set
# ...
# 
estate=pd.read_csv('Real_estate.csv',index_col='No',dtype=float)
estate=estate.sample(frac=1).reset_index(drop=True)
split_at=int(0.3*(estate.shape[0]))
X_train=estate.iloc[:split_at,:-1]
y_train=estate.iloc[:split_at,-1]
X_test=estate.iloc[split_at:,:-1]
y_test=estate.iloc[split_at:,-1]
      
model=DecisionTree(max_depth=2)
model.fit(X_train,y_train)
y_out=model.predict(X_test)
print("Rmse is: ",rmse(y_out,y_test))
print("Mae is: ",mae(y_out,y_test))

model2=DecisionTreeRegressor(max_depth=2)
model2.fit(X_train,y_train)
y_out=model2.predict(X_test)
print("Rmse of Sklearn is: ",rmse(y_out,y_test))
print("Mae of Sklearn is: ",mae(y_out,y_test))
