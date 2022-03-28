import numpy as np

# X is the input, it basically has 2 features
X = np.array(([2, 9], [1, 5], [3, 6]), dtype = float)
y = np.array(([92], [86], [89]), dtype = float)
X = X/np.amax(X,axis=0) # Used to normalize the data. Each number is divided by 9, since it is the max
y = y/100 

# We are trying to get from X to Y
# X = [0.222, 1], [0.111, 0.555]
# Y = [0.92], [0.86]

# Sigmoid Function - it is the activation function
def sigmoid (x):
    return 1/(1 + np.exp(-x))

# Derivative of Sigmoid Function
def derivatives_sigmoid(x):
    return x * (1 - x)

# Variable initialization
epoch = 5 # Setting training iterations
lr = 0.1 # Setting learning rate

# number of features in data set
# It is 2 because each list in X has 2 elements
inputlayer_neurons = 2 
hiddenlayer_neurons = 3
# number of neurons at output layer
# It is 1, because each list in Y has one element
output_neurons = 1 

# Weight of hidden layer (input -> hidden)
wh=np.random.uniform(size=(inputlayer_neurons, hiddenlayer_neurons))
# Bias of hidden layer 
bh=np.random.uniform(size=(1, hiddenlayer_neurons))
# Weight of output layer (hidden -> ouput)
wout=np.random.uniform(size=(hiddenlayer_neurons, output_neurons))
# Bias of output layer
bout=np.random.uniform(size=(1,output_neurons))

# draws a random range of numbers uniformly of dim x*y
for i in range(epoch):
    # Forward Propogation
    # -----------------
    # Get the dot product of X and the weights of each hidden layer (wh)
    hinp1=np.dot(X,wh)
    # Add the bias (bh)
    hinp=hinp1 + bh
    # Get the activation of hidden layer using previous input (hinp)
    hlayer_act = sigmoid(hinp)
    # Get the input for the output layer, and multiply it with output weights (wout)
    outinp1 = np.dot(hlayer_act,wout)
    # Add the bias to this (bout)
    outinp = outinp1+bout
    # Activate the output of the output layer
    output = sigmoid(outinp)
    
    # Backpropagation
    # -------------------
    # EO = Error of our calculated output. Calculate it by doing Y - output
    EO = y-output
    # Calculate gradient of output using sigmoid derivative
    outgrad = derivatives_sigmoid(output)
    # Mutliply it with error
    d_output = EO * outgrad
    # Error in Hidden layer
    EH = d_output.dot(wout.T)
    # Gradient of hidden layer using sigmoid derivative
    hiddengrad = derivatives_sigmoid(hlayer_act) 
    d_hiddenlayer = EH * hiddengrad
    
    wout += hlayer_act.T.dot(d_output) * lr   # dotproduct of nextlayererror and currentlayerop
    wh += X.T.dot(d_hiddenlayer) * lr
    
    print ("-----------Epoch-", i+1, "Starts----------")
    print("Input: \n" + str(X)) 
    print("Actual Output: \n" + str(y))
    print("Predicted Output: \n" ,output)
    print ("-----------Epoch-", i+1, "Ends----------\n")
        
print("Input: \n" + str(X)) 
print("Actual Output: \n" + str(y))
print("Predicted Output: \n" ,output)