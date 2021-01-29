"""
The current code given is for the Assignment 1.
You will be expected to use this to make trees for:
> discrete input, discrete output
> real input, real output
> real input, discrete output
> discrete input, real output
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from .utils import entropy, information_gain, gini_index
np.random.seed(42)

class Node():
    def __init__(self):
        self.type=None              #Here type is: discrete and real
        self.feature=None           #feature is the column to concider
        self.split_val=None         #Spliting number for Real input
        self.value=None             #Output value
        self.child={}               #Contains children nodes
        self.depth=None             #Stores the node depth
        self.isleaf=False

class DecisionTree():
    def __init__(self, criterion="gini_index", max_depth=10):
        """
        Put all infromation to initialize your tree here.
        Inputs:
        > criterion : {"information_gain", "gini_index"} # criterion won't be used for regression
        > max_depth : The maximum depth the tree can grow to 
        """
        self.Tree=None
        self.criterion=criterion
        self.max_depth=max_depth

    def find_split(self,X_features,y,type):
        Feature=0
        split_value=None
        if(type=='DD'):
            opt_loss=-10000000000
            y_size=y.size
            for i in X_features:
                column=X_features[i]
                loss_value=None
                if(self.criterion=="information_gain"):
                    loss_value=information_gain(y,column)
                else:
                    groups=np.unique(column)
                    gini=0
                    for j in groups:
                        y_temp=pd.Series([y[idx] for idx in range(y.size) if column[idx]==j])
                        gini+=gini_index(y_temp)*(y_temp.size)
                    loss_value=gini*(-1/y_size)
                if(loss_value>opt_loss):
                    Feature=i
                    opt_loss=loss_value

        elif(type=='RD'):
            opt_loss=-10000000000
            y_size=y.size
            for i in X_features:
                column=X_features[i]
                sorted_col=list(column.sort_values())
                loss=None
                for j in range(len(sorted_col)-1):
                    spliting_value=(sorted_col[j]+sorted_col[j+1])/2
                    if(self.criterion=="information_gain"):
                        mask_attr=np.where(column<=spliting_value,2,4)
                        loss=information_gain(y,mask_attr)
                    else:
                        y_left=pd.Series([y[idx] for idx in range(y_size) if column[idx]<=spliting_value])
                        y_right=pd.Series([y[idx] for idx in range(y_size) if column[idx]>spliting_value])
                        loss=((y_left.size)*gini_index(y_left)+(y_right.size)*gini_index(y_right))*(-1/y_size)
                    if(loss>opt_loss):
                        opt_loss=loss
                        Feature=i
                        split_value=spliting_value    

        elif(type=='DR'):
            opt_loss=1000000000
            y_size=y.size
            for i in X_features:
                column=X_features[i]
                loss=0
                groups=np.unique(column)
                for j in groups:
                    y_temp=pd.Series([y[idx] for idx in range(y_size) if column[idx]==j])
                    loss+=(y_temp.size)*np.var(y_temp)
                if(opt_loss>loss):
                    Feature=i
                    opt_loss=loss            

        else:
            opt_loss=10000000000
            y_size=y.size
            for i in X_features:
                column=X_features[i]
                col_list=list(column.sort_values())
                loss=None
                for j in range(y_size-1):
                    spliting_value=(col_list[j]+col_list[j+1])/2
                    y_left=pd.Series([y[idx] for idx in range(y_size) if column[idx]<=spliting_value])
                    y_right=pd.Series([y[idx] for idx in range(y_size) if column[idx]>spliting_value])
                    loss=y_left.size*np.var(y_left) + y_right.size*np.var(y_right)
                    if(loss<opt_loss):
                        opt_loss=loss
                        Feature=i
                        split_value=spliting_value

        return Feature,split_value
 
    def fit_Tree(self,X,y,depth):
        node=Node()
        split_value=None
        Feature=None
        if(y.dtype.name=="category"):
            groups=np.unique(y)
            a=list(y)
            if(X.shape[1]==0 or self.max_depth==depth or groups.size==1):
                node.isleaf=True
                node.value=max(set(a),key=a.count)
                node.type='D'
                node.depth=depth
                return node   
            if(not('float' in list(X.dtypes))):
                Feature,split_value=self.find_split(X,y,'DD')
            else:
                Feature,split_value=self.find_split(X,y,'RD')
        else:
            if(X.shape[1]==0 or self.max_depth==depth or y.size==1):
                node.isleaf=True
                node.type='R'
                node.value=y.mean()
                node.depth=depth
                return node
            if(not('float' in list(X.dtypes))):
                Feature,split_value=self.find_split(X,y,'DR')
            else:
                Feature,split_value=self.find_split(X,y,'RR')

        if(split_value==None):
            node.type='D'
            node.feature=Feature
            node.depth=depth
            groups=np.unique(X[Feature])
            for i in groups:
                X_split=X[X[Feature]==i].reset_index().drop(['index',Feature],axis=1)
                y_split=pd.Series([y[j] for j in range(y.size) if X[Feature][j]==i],dtype=y.dtype)
                node.child[i]=self.fit_Tree(X_split,y_split,depth+1)                        
        else:
            node.type='R'
            node.feature=Feature
            node.depth=depth
            node.split_val=split_value
            X_left=X[X[Feature]<=split_value].reset_index(drop=True)
            y_left=pd.Series([y[j] for j in range(y.size) if X[Feature][j]<=split_value],dtype=y.dtype)
            X_right=X[X[Feature]>split_value].reset_index(drop=True)
            y_right=pd.Series([y[j] for j in range(y.size) if X[Feature][j]>split_value],dtype=y.dtype)
            node.child["left"]=self.fit_Tree(X_left,y_left,depth+1)
            node.child["right"]=self.fit_Tree(X_right,y_right,depth+1)
        return node

    def fit(self, X, y):
        """
        Function to train and construct the decision tree
        Inputs:
        X: pd.DataFrame with rows as samples and columns as features (shape of X is N X P) where N is the number of samples and P is the number of columns.
        y: pd.Series with rows corresponding to output variable (shape of Y is N)
        """
        self.Tree=self.fit_Tree(X,y,0)

    def predict(self, X):
        """
        Funtion to run the decision tree on a data point
        Input:
        X: pd.DataFrame with rows as samples and columns as features
        Output:
        y: pd.Series with rows corresponding to output variable. THe output variable in a row is the prediction for sample in corresponding row in X.
        """
        output=list()
        for rowidx in range(X.shape[0]):
            row=X.iloc[rowidx]
            DTree=self.Tree
            while(not DTree.isleaf) :
                if(DTree.type=='R'):
                    if(row[DTree.feature]<=DTree.split_val):
                        DTree=DTree.child["left"]
                    else:
                        DTree=DTree.child["right"]
                else:
                    DTree=DTree.child[row[DTree.feature]]            
            output.append(DTree.value)        
        return pd.Series(output)      

    def plot_tree(self,Tnode):
        if(Tnode.isleaf):
            print("Value is := ",Tnode.value)
        else:
            if(Tnode.type=='R'):
                print("?(X{} > {})".format(Tnode.feature,Tnode.split_val))
                print("\t"*(Tnode.depth+1),"Y:= ",end="")
                self.plot_tree(Tnode.child["left"])
                print("\t"*(Tnode.depth+1),"N:= ",end="")  
                self.plot_tree(Tnode.child["right"])
            else:
                flag=0
                ma=len(Tnode.child)
                for i in Tnode.child:
                    flag+=1
                    print("?(X{} == {})".format(Tnode.feature,i))
                    print("\t"*(Tnode.depth+1),end="")
                    self.plot_tree(Tnode.child[i])    
                    if(flag<ma):
                        print("\t"*Tnode.depth,end="")

    def plot(self):
        """
        Function to plot the tree

        Output Example:
        ?(X1 > 4)
            Y: ?(X2 > 7)
                Y: Class A
                N: Class B
            N: Class C
        Where Y => Yes and N => No
        """
        self.plot_tree(self.Tree)
