import numpy as np

class Layer:
    def __init__(self):
        self.size = None
        self.input_size = None
    def compile(self, input_size):
        pass
    def forward(self, input):
        pass
    def backward(self, output_gradient, learning_rate):
        pass
class Input(Layer):
    def __init__(self, size):
        self.size = size
    def compile(self, input_size):
        pass
    def forward(self, input):
        return input
    def backward(self, output_gradient, learning_rate):
        return output_gradient
class Dense(Layer):
    def __init__(self, size, activation):
        self.size = size
        self.activation = activation
    def compile(self, input_size):
        self.bias = np.random.randn(self.size, 1)
        self.weights = np.random.randn(self.size, input_size)
    def forward(self, input):
        self.input = input
        self.activation_input = np.dot(self.input[0], self.weights.T) + self.bias.T
        return self.activation.function(self.activation_input)
    def backward(self, output_gradient, learning_rate):
        output_gradient = np.multiply(output_gradient, self.activation.derivative(self.activation_input))
        weights_gradient = np.dot(np.array(self.input).T, np.array(output_gradient))
        self.weights -= learning_rate * weights_gradient.T
        self.bias -= learning_rate * output_gradient.T
        return np.dot(output_gradient, self.weights)
class Flatten(Layer):
    def __init__(self, input_shape):
        self.input_shape = input_shape
    def compile(self):
        self.size = self.input_shape[0] * self.input_shape[0]
    def forward(self, input):
        return np.reshape(input, (1, -1))
    def backward(self, output_gradient, learning_rate):
        return np.reshape(output_gradient, self.input_shape)