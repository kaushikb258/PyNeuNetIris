"""
-----------------------------------------
 NEURAL NETWORK WITH BACKPROPAGATION
 AUTHOR: KAUSHIK BALAKRISHNAN, PHD
 kaushikb258@gmail.com
-----------------------------------------
"""

import pandas as pd
import numpy as np

def read_raw_data(ntrain,ntest,ninputs,noutputs):
     
     d = pd.read_csv("iris",header=None)
     # shuffle data
     d = d.reindex(np.random.permutation(d.index))
     
     train_in = np.zeros((ntrain,ninputs),dtype=np.float64)
     train_out = np.zeros((ntrain,noutputs),dtype=np.float64)
     test_in = np.zeros((ntest,ninputs),dtype=np.float64)
     test_out = np.zeros((ntest,noutputs),dtype=np.float64)

     d = np.array(d)  
       
     for j in range(ninputs):
      for i in range(ntrain):         
       train_in[i,j] = d[i,j]
      for i in range(ntest): 
       test_in[i,j] = d[ntrain-1+i,j]

     for i in range(ntrain):  
       if(d[i,ninputs] == "Iris-setosa"):
        train_out[i,:] = [1, 0, 0]
       elif(d[i,ninputs] == "Iris-versicolor"):
        train_out[i,:] = [0, 1, 0]
       elif(d[i,ninputs] == "Iris-virginica"):
        train_out[i,:] = [0, 0, 1]
       else:
        print "ERROR IN READING DATA"         

     for i in range(ntest):  
       if(d[ntrain-1+i,ninputs] == "Iris-setosa"):
        test_out[i,:] = [1, 0, 0]
       elif(d[ntrain-1+i,ninputs] == "Iris-versicolor"):
        test_out[i,:] = [0, 1, 0]
       elif(d[ntrain-1+i,ninputs] == "Iris-virginica"):
        test_out[i,:] = [0, 0, 1]
       else:
        print "ERROR IN READING DATA"         


     return train_in, train_out, test_in, test_out   

#---------------------------------     