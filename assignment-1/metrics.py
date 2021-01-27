import numpy as np
def accuracy(y_hat, y):
    """
    Function to calculate the accuracy

    Inputs:
    > y_hat: pd.Series of predictions
    > y: pd.Series of ground truth
    Output:
    > Returns the accuracy as float
    """
    """
    The following assert checks if sizes of y_hat and y are equal.
    Students are required to add appropriate assert checks at places to
    ensure that the function does not fail in corner cases.
    """
    assert(y_hat.size == y.size)
    # TODO: Write here
    pridct=list(y_hat)
    Gtruth=list(y)
    correct_pridiction=0
    total=len(pridct)
    for i in range(len(pridct)):
        if(pridct[i]==Gtruth[i]):
            correct_pridiction+=1
    return correct_pridiction/total        

def precision(y_hat, y, cls):
    """
    Function to calculate the precision

    Inputs:
    > y_hat: pd.Series of predictions
    > y: pd.Series of ground truth
    > cls: The class chosen
    Output:
    > Returns the precision as float
    """
    predict=list(y_hat)
    Gtruth=list(y)
    Trueclass=0
    allclass=0
    for i in range(len(predict)):
        if(predict[i]==cls):
            if(predict[i]==Gtruth[i]):
                Trueclass+=1
        allclass+=1
    return Trueclass/allclass            

def recall(y_hat, y, cls):
    """
    Function to calculate the recall

    Inputs:
    > y_hat: pd.Series of predictions
    > y: pd.Series of ground truth
    > cls: The class chosen
    Output:
    > Returns the recall as float
    """
    predict=list(y_hat)
    Gtruth=list(y)
    Trueclass=0
    allclass=0
    for i in range(len(predict)):
        if(Gtruth[i]==cls):
            if(predict[i]==Gtruth[i]):
                Trueclass+=1
        allclass+=1
    return Trueclass/allclass


def rmse(y_hat, y):
    """
    Function to calculate the root-mean-squared-error(rmse)
    Inputs:
    > y_hat: pd.Series of predictions
    > y: pd.Series of ground truth
    Output:
    > Returns the rmse as float
    """
    predict=np.array(y_hat)
    Gtruth=np.array(y)
    return (np.sqrt(np.mean((predict-Gtruth)**2)))

def mae(y_hat, y):
    """
    Function to calculate the mean-absolute-error(mae)

    Inputs:
    > y_hat: pd.Series of predictions
    > y: pd.Series of ground truth
    Output:
    > Returns the mae as float
    """
    predict=np.array(y_hat)
    Gtruth=np.array(y)
    return (np.mean(abs(predict-Gtruth)))
