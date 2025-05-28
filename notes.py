# Neural Nets, Chapter 1

# Overfitting, when the network just memorizes the training data and cannot generalize!

"""
You will then train your model with the other 90,000 in-sample or “training”
data and finally validate your model with the 10,000 out-of-sample data that the model hasn’t yet
seen. The goal is for the model to not only accurately predict on the training data, but also to be
similarly accurate while predicting on the withheld out-of-sample validation data.
"""

# Neural Nets, Chapter 2
# Coding our first neuron

"""
Inputs are the data that we pass into the model
to get desired outputs, while the weights are the parameters that we’ll tune later on to get these
results.
"""

"""
Python 3.7.5
NumPy 1.15.0
Matplotlib 3.1.1
"""

inputs = [1, 2, 3]
weigths = [0.2, 0.8, -0.5]
bias = 2

# The equation is 1*0.2 + 2*0.8 + 3*-0.5 + 2, for a single neuron
output = (inputs[0]*weights[0] + inputs[1]*weights[1] + inputs[2]*weights[2] + bias)

print(output)
