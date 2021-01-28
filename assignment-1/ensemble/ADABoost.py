import numpy as np
from sklearn.tree import DecisionTreeClassifier
class AdaBoostClassifier():
    def __init__(self, base_estimator, n_estimators=3): # Optional Arguments: Type of estimator
        '''
        :param base_estimator: The base estimator model instance from which the boosted ensemble is built (e.g., DecisionTree, LinearRegression).
                               If None, then the base estimator is DecisionTreeClassifier(max_depth=1).
                               You can pass the object of the estimator class
        :param n_estimators: The maximum number of estimators at which boosting is terminated. In case of perfect fit, the learning procedure may be stopped early.
        '''
        self.n_estimators=n_estimators
        if(base_estimator==None):
            self.base_estimator=DecisionTreeClassifier(max_depth=1)
        else:    
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
            Dt=self.base_estimator
            Dt.fit(X,y,sample_weight=weight)
            mask=y!=np.array(Dt.predict(X))
            error=np.sum(weight*mask)/np.sum(weight)
            alpha=0
            if(error!=0):
                alpha=np.log((1-error)/error)  
            weight=weight*np.exp(np.where(mask,1,-1)*alpha)
            self.trees[i]=(alpha,Dt)

    def predict(self, X):
        """
        Input:
        X: pd.DataFrame with rows as samples and columns as features
        Output:
        y: pd.Series with rows corresponding to output variable. THe output variable in a row is the prediction for sample in corresponding row in X.
        """
        y=0
        for i in range(self.n_estimators):
            alpha,Dt=self.trees[i]
            y+=alpha*Dt.predict(X)
        y=np.where(y!=0,abs(y)/y,1)
        return y

    def plot(self):
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
        pass
