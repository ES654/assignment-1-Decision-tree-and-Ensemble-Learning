import math
def entropy(Y):
    """
    Function to calculate the entropy 
    Inputs:
    > Y: pd.Series of Labels
    Outpus:
    > Returns the entropy as a float
    """
    values=list(Y)
    total_size=len(Y)
    group=dict()
    for i in values:
        if i in group:
            group[i]+=1
        else:
            group[i]=1
    etr=0
    for i in group:
        prb=group[i]/total_size
        etr-=prb*math.log2(prb)
    return etr

def gini_index(Y):
    """
    Function to calculate the gini index
    Inputs:
    > Y: pd.Series of Labels
    Outpus:
    > Returns the gini index as a float
    """
    values=list(Y)
    total_size=len(Y)
    group=dict()
    for i in values:
        if i in group:
            group[i]+=1
        else:
            group[i]=1
    gini=1
    for i in group:
        prb=group[i]/total_size
        gini-=prb**2
    return gini


def information_gain(Y, attr):
    """
    Function to calculate the information gain
    Inputs:
    > Y: pd.Series of Labels
    > attr: pd.Series of attribute at which the gain should be calculated
    Outputs:
    > Return the information gain as a float
    """
    group=dict()
    Y=list(Y)
    total_size=len(Y)
    attr=list(attr)
    for i in range(len(attr)):
        if attr[i] in group:
            group[attr[i]].append(Y[i])
        else:
            group[attr[i]]=list(Y[i])
    IG=entropy(Y)
    for i in group:
        IG-=(len(group[i])/total_size)*entropy(group[i])
    return IG  
