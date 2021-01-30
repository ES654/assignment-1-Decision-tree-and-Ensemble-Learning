from sklearn.tree import DecisionTreeClassifier
import pandas as pd
import numpy as np
from sklearn import tree
import matplotlib.pyplot as plt
class BaggingClassifier():
    def __init__(self, base_estimator=DecisionTreeClassifier(), n_estimators=100):
        '''
        :param base_estimator: The base estimator model instance from which the bagged ensemble is built (e.g., DecisionTree(), LinearRegression()).
                               You can pass the object of the estimator class
        :param n_estimators: The number of estimators/models in ensemble.
        '''
        self.n_estimators=n_estimators
        self.base_estimator=base_estimator   
        self.trees=[None]*self.n_estimators

    def Unif_sample(self,X,y):
        temp=pd.DataFrame(X)
        y_g=list(y)
        length=temp.shape[0]
        y_samp=[0]*length
        for i in range(length):
            k=np.random.randint(length)
            temp.iloc[i]=list(X.iloc[k])
            y_samp[i]=y_g[k]
        return temp,y_samp 

    def fit(self, X, y):
        """
        Function to train and construct the BaggingClassifier
        Inputs:
        X: pd.DataFrame with rows as samples and columns as features (shape of X is N X P) where N is the number of samples and P is the number of columns.
        y: pd.Series with rows corresponding to output variable (shape of Y is N)
        """
        for i in range(self.n_estimators):
            modl=DecisionTreeClassifier()
            a,b=self.Unif_sample(X,y)
            modl.fit(a,b)
            self.trees[i]=modl
            
    def predict(self, X):
        """
        Funtion to run the BaggingClassifier on a data point
        Input:
        X: pd.DataFrame with rows as samples and columns as features
        Output:
        y: pd.Series with rows corresponding to output variable. THe output variable in a row is the prediction for sample in corresponding row in X.
        """
        len=X.shape[0]
        predicts=dict()
        Y_pred=[0]*len
        for i in range(self.n_estimators):
            predicts[i]=self.trees[i].predict(X)
        for i in range(len):
            a=list()
            for j in range(self.n_estimators):
                a.append(predicts[j][i])
            Y_pred[i]=max(set(a),key=a.count)        
        return pd.Series(Y_pred)

    def plot(self,X):
        """
        Function to plot the decision surface for BaggingClassifier for each estimator(iteration).
        Creates two figures
        Figure 1 consists of 1 row and `n_estimators` columns and should look similar to slide #16 of lecture
        The title of each of the estimator should be iteration number

        Figure 2 should also create a decision surface by combining the individual estimators and should look similar to slide #16 of lecture

        Reference for decision surface: https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html

        This function should return [fig1, fig2]
        """
        for i in range(self.n_estimators):
            tree.plot_tree(self.trees[i])
            temp="Tree number"+str(i)
            plt.title(temp)
            plt.show()

        x_min, x_max = X.iloc[:, 0].min() - 1, X.iloc[:, 0].max() + 1
        y_min, y_max = X.iloc[:, 1].min() - 1, X.iloc[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),np.arange(y_min, y_max, 0.02))
        plt.tight_layout(h_pad=0.5, w_pad=0.5, pad=2.5)
        Z = np.array(self.predict(np.c_[xx.ravel(), yy.ravel()]))
        Z = Z.reshape(xx.shape)
        cs = plt.contourf(xx, yy, Z, cmap=plt.cm.RdYlBu)

        y_hat=list(self.predict(X))
        x_axis=list(X.iloc[:,0])
        y_axis=list(X.iloc[:,1])
        for i in range(len(x_axis)):
            if(y_hat[i]==1):
                plt.scatter(x_axis[i],y_axis[i],c='RED',cmap=plt.cm.RdYlBu)
            else:
                plt.scatter(x_axis[i],y_axis[i],c='BLUE',cmap=plt.cm.RdYlBu)
        plt.show()
