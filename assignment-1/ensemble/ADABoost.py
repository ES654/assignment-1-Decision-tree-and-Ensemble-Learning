from matplotlib import pyplot as plt
import numpy as np
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
class AdaBoostClassifier():
    def __init__(self, base_estimator=DecisionTreeClassifier(max_depth=1), n_estimators=3): # Optional Arguments: Type of estimator
        '''
        :param base_estimator: The base estimator model instance from which the boosted ensemble is built (e.g., DecisionTree, LinearRegression).
                               If None, then the base estimator is DecisionTreeClassifier(max_depth=1).
                               You can pass the object of the estimator class
        :param n_estimators: The maximum number of estimators at which boosting is terminated. In case of perfect fit, the learning procedure may be stopped early.
        '''
        self.n_estimators=n_estimators
        self.base_estimator=base_estimator
        self.trees=[None]*n_estimators

    def fit(self, X, y):
        """
        Function to train and construct the AdaBoostClassifier
        Inputs:
        X: pd.DataFrame with rows as samples and columns as features (shape of X is N X P) where N is the number of samples and P is the number of columns.
        y: pd.Series with rows corresponding to output variable (shape of Y is N)
        """
        y=np.array(y)
        size_y=len(y)
        weight=np.array([1/size_y]*size_y)
        for i in range(self.n_estimators):
            Dt=DecisionTreeClassifier(max_depth=1)
            Dt.fit(X,y,sample_weight=weight)
            mask=y!=np.array(Dt.predict(X))
            error=np.sum(weight*mask)/np.sum(weight)
            alpha=0
            if(error!=0):
                alpha=(0.5)*np.log((1-error)/error)  
            weight=weight*np.exp(np.where(mask,1,-1)*alpha)
            self.trees[i]=(alpha,Dt,weight)

    def predict(self, X):
        """
        Input:
        X: pd.DataFrame with rows as samples and columns as features
        Output:
        y: pd.Series with rows corresponding to output variable. THe output variable in a row is the prediction for sample in corresponding row in X.
        """
        y=np.zeros(X.shape[0])
        for i in range(self.n_estimators):
            alpha,Dt,weight=self.trees[i]
            y+=alpha*Dt.predict(X)
        y=np.where(y<=0,-1,1)
        return pd.Series(y)

    def plot(self,X):
        """
        Function to plot the decision surface for AdaBoostClassifier for each estimator(iteration).
        Creates two figures
        Figure 1 consists of 1 row and `n_estimators` columns
        The title of each of the estimator should be associated alpha (similar to slide#38 of course lecture on ensemble learning)
        Further, the scatter plot should have the marker size corresponnding to the weight of each point.

        Figure 2 should also create a decision surface by combining the individual estimators

        Reference for decision surface: https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html

        This function should return [fig1, fig2]
        """
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

        # fig,ax=plt.subplots(1,self.n_estimators)
        # for i in range(self.n_estimators):
        #     alpha,Dt,weights=self.trees[i]
        #     y_hat=list(Dt.predict(X))
        #     weight=list(weights/np.max(weights)*40)
        #     x_axis=list(X.iloc[:,0])
        #     y_axis=list(X.iloc[:,1])
        #     print(len(weight),len(x_axis),len(y_axis))
        #     ax[i].scatter(x_axis,y_axis,s=weight)
        #     temp="Alpha value is: "+str(alpha)
        #     ax[i].set_title(temp)
        # plt.show()
