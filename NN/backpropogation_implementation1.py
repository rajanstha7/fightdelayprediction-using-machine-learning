import numpy as np
from preprocess_Data1 import features, targets, features_test, targets_test


np.random.seed(42)

def sigmoid(x):
    """Calculate sigmoid"""
    return 1 / (1 + np.exp(-x))


# hyperparameters
n_hidden = 3 # number of hidden units
epochs = 10
learning_rate = 0.5

#n_examples, n_features =  len(targets), 
n_examples, n_features =  np.shape(features) 


# TODO    ### Here n_examples means number of training examples, n_features means features of our input, fill in the appropriate values as it suits our problem


last_loss = None
# initialize weights
weights_input_hidden = np.random.normal(scale=1 / n_features ** -.5, size=(n_features, n_hidden))
print(weights_input_hidden)
# TODO: Replace val1, and val2 with appropriate values, try not to hardcode values, refer to np.random.normal documentation for help
weights_hidden_output =np.random.normal(scale=1 / n_features ** -.5, size=n_hidden) # TODO: Replace val3 with appropriate values, try not to hardcode values and instead find a generic way
print(weights_hidden_output)
                                         

for e in range(epochs):
    del_w_input_hidden = np.zeros(weights_input_hidden.shape)
    del_w_hidden_output = np.zeros(weights_hidden_output.shape) # TODO: Something very similar to the expression above, fill in yourself

    for x, y in zip(features.values, targets):
        # forward pass
        # calculate the output
        hidden_input = np.dot(x, weights_input_hidden)
        hidden_activations = sigmoid(hidden_input)
        
        
        output_layer_input = np.dot(hidden_activations, weights_hidden_output)
        
        # TODO: Fill this, similar to hidden_input above. For help look at backpropogation.py script from last time
        output = sigmoid(output_layer_input)
        
        # backward pass
        # calculate the error
        error = y - output

        # calculate error gradient in output unit
        output_error = error * output * (1 - output)

        # propagate errors to hidden layer
        hidden_error = np.dot(output_error,  weights_hidden_output) * hidden_activations * (1 - hidden_activations)
        # TODO: Fill this, similar to output_error above. For help look at backpropogation.py script from last time

        # update the change in weights
        del_w_hidden_output += output_error * hidden_activations
        del_w_input_hidden +=   hidden_error * x[:, None] 
        # TODO: Fill this yourself. For help look at backpropogation.py script from last time

    
    # update weights
    weights_hidden_output += learning_rate * del_w_hidden_output / n_examples
    weights_input_hidden += learning_rate * del_w_input_hidden / n_examples
    # TODO: Fill this yourself. For help look at backpropogation.py script from last time

    # printing out the mean square error on the training set
    if e % (epochs / 10) == 0:
        hidden_activations = sigmoid(np.dot(x, weights_input_hidden))
        out = sigmoid(np.dot(hidden_activations,
                             weights_hidden_output))
        loss = np.mean((out - targets) ** 2)

        if last_loss and last_loss < loss:
            print("Train loss: ", loss, "  WARNING - Loss Increasing")
        else:
            print("Train loss: ", loss)
        last_loss = loss              
        

# calculate accuracy on test data
hidden = sigmoid(np.dot(features_test, weights_input_hidden))
out = sigmoid(np.dot(hidden, weights_hidden_output))


predictions = out > 0.5
accuracy = np.mean(predictions == targets_test)
print("Prediction accuracy: {:.3f}".format(accuracy))          


####################################################################################################################################################
## Task 2: If you complete the above task successfully, try to remove the "for loop" above and vectorize the process for more efficient computation.
####################################################################################################################################################
            
