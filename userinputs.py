"""
-----------------------------------------
 NEURAL NETWORK WITH BACKPROPAGATION
 AUTHOR: KAUSHIK BALAKRISHNAN, PHD
 kaushikb258@gmail.com
-----------------------------------------
"""

#------------------------------------------------
def set_inputs():

# number of hidden layers
 nhidden = 3 

# maximum number of neurons per layer
 max_neurons = 6 

# weight adjustment factor
 beta = 0.5

# number of Neural Net iterations
 niters = 2000

# number of inputs/features
 ninputs = 4

# number of outputs/targets
 noutputs = 3

# number of training samples
 ntrain = 100

# number of test samples
 ntest = 50

# update procedure
 update_procedure = 1

# update_procedure = 1 for the classical beta approach

# number of neurons per layer
 num_neurons = [ninputs, 4, 3, 4, noutputs] 

 return nhidden, max_neurons, beta, niters, ninputs, noutputs, ntrain, ntest, update_procedure, num_neurons
