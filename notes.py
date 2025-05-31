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


inputs = [1.0, 2.0, 3.0, 2.5]
weights = [0.2, 0.8, -0.5, 1.0]
bias = 2.0

# The function if we add on more weight and input
output = (inputs[0]*weights[0] + inputs[1]*weights[1] + inputs[2]*weights[2] + inputs[3]*weights[3] + bias)

# A layer of neurons

inputs = [1, 2, 3, 2.5]
weights1 = [0.2, 0.8, -0.5, 1]
weights2 = [0.5, -0.91, 0.26, -0.5]
weights3 = [-0.26, -0.27, 0.17, 0.87]
bias1 = 2
bias2 = 3
bias3 = 0.5
outputs = [
# Neuron 1:
inputs[0]*weights1[0] +
inputs[1]*weights1[1] +
inputs[2]*weights1[2] +
inputs[3]*weights1[3] + bias1,
# Neuron 2:
inputs[0]*weights2[0] +
inputs[1]*weights2[1] +
inputs[2]*weights2[2] +
inputs[3]*weights2[3] + bias2,
# Neuron 3:
inputs[0]*weights3[0] +
inputs[1]*weights3[1] +
inputs[2]*weights3[2] +
inputs[3]*weights3[3] + bias3]
print(outputs)
