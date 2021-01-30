
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
from tree.base import DecisionTree
from metrics import *
from time import time
np.random.seed(42)
num_average_time = 100

# Learn DTs 
# ...
# 
# Function to calculate average time (and std) taken by fit() and predict() for different N and P for 4 different cases of DTs
# Function to plot the results
# ..
# Function to create fake data (take inspiration from usage.py)
def CreateFakeData(N,P,type):
    if(type=='DD'):
        X = pd.DataFrame({i:pd.Series(np.random.randint(P, size = N), dtype="category") for i in range(5)})
        y = pd.Series(np.random.randint(P, size = N), dtype="category")
        return X,y
    elif(type=='RD'):
        X = pd.DataFrame(np.random.randn(N, P))
        y = pd.Series(np.random.randint(P, size = N), dtype="category")
        return X,y
    elif(type=='DR'):
        X = pd.DataFrame({i:pd.Series(np.random.randint(P, size = N), dtype="category") for i in range(5)})
        y = pd.Series(np.random.randn(N))
        return X,y
    else:
        X = pd.DataFrame(np.random.randn(N, P))
        y = pd.Series(np.random.randn(N))
        return X,y

def find_Time(case):
    axis_Nf=[0]*25 # These are time for different N values by fixing P on fit
    axis_Np=[0]*25 # for predict
    axis_Pf=[0]*11 # Different P for fixed P and for model fit
    axis_Pp=[0]*11 # for predict
    print("Started 1")
    for i in range(100,500,20):
        X,y=CreateFakeData(i,5,case) #we fix p = 5
        mod=DecisionTree()
        st1=time()
        mod.fit(X,y)
        ed1=time()
        st2=time()
        y_=mod.predict(X)
        ed2=time()
        axis_Nf[(i-100)//20]=(ed1-st1)
        axis_Np[(i-150)//20]=(ed2-st2)
    print("Started 2")    
    for i in range(2,24,2):
        X,y=CreateFakeData(100,i,case)
        mod=DecisionTree()
        st1=time()
        mod.fit(X,y)
        ed1=time()
        st2=time()
        y_=mod.predict(X)
        ed2=time()
        axis_Pf[(i-2)//2]=(ed1-st1)
        axis_Pp[(i-2)//2]=(ed2-st2)
    return axis_Nf,axis_Np,axis_Pf,axis_Pp

def plot_points(case):
    Y1,Y2,Y3,Y4=find_Time(case)
    X1=list(range(10,500,20))
    X2=list(range(2,24,2))
    fig,ax=plt.subplots(2,2)
    ax[0,0].plot(X1,Y1)
    ax[0,0].set_xlabel("Different N for fixed P=5")
    ax[0,0].set_title("Fit time in Y-axis")
    ax[0,1].plot(X1,Y2)
    ax[0,1].set_xlabel("Different N for fixed p=5")
    ax[0,1].set_title("Predict time in Y-axis")
    ax[1,0].plot(X2,Y3)
    ax[1,0].set_xlabel("Different P for fixed N=100")
    ax[1,0].set_title("Fit time in Y-axis")
    ax[1,1].plot(X2,Y4)
    ax[1,1].set_xlabel("Different P for fixed N=100")
    ax[1,1].set_title("Predict time in Y-axis")
    temp="This is for case: "+case
    fig.suptitle(temp)
    plt.subplots_adjust(wspace=0.5,hspace=0.5)
    plt.show()

plot_points('DD')
plot_points('DR')
plot_points('RD')
plot_points('RR')
# ...
# ..other functions
