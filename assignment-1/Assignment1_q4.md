# Visualizing Time complexity by varying P and N 

1.  **Discrete Input and Discrete Output case:-**  
    >![alt text](DD.png "DD Case")

2. **Discrete Input and Real Output case:-** 
    >![alt text](DR.png "DR Case")

3. **Real Input and Discrete Output case:-**
    >![alt text](RD.png "RD Case")

4. **Real Input and Real Output case:-**
    >![alt text](RR.png "RR Case")

* Here we can see that the Real inputs takes a lot of time to compute, Since we have to check all the splits.
* Here the the graph of N vs T is of order O(N<sup>2</sup>) for Real input and its O(N) for discrete input().
* Same applies for P vs T.
* When it comes to Predict its completely unpridictable and depends on Tree depth and How many times a feature is checked.
* Again Real output cases have Large Trees, which make them consume a little bit extra time.
