import numpy as np

from NeuralNetwork.Layers.Layer import Layer
from NeuralNetwork.Activations.Activation import Activation

class Dense(Layer):
    def __init__(self, shape: int, activation: Activation) -> None:
        self.output_shape = shape
        self.activation = activation
    def compile(self, input_shape: int) -> None:
        self.bias = np.random.randn(self.output_shape, 1)
        self.weights = np.random.randn(self.output_shape, input_shape)
    def forward(self, input: np.ndarray) -> np.ndarray:
        self.input = input
        self.activation_input = np.dot(self.input[0], self.weights.T) + self.bias.T
        return self.activation.function(self.activation_input)
    def backward(self, output_gradient: np.ndarray, learning_rate: float) -> np.ndarray:
        output_gradient = np.multiply(output_gradient, self.activation.derivative(self.activation_input))
        self.weights_gradient = np.dot(np.array(self.input).T, np.array(output_gradient))
        self.weights -= learning_rate * self.weights_gradient.T
        self.bias -= learning_rate * output_gradient.T
        return np.dot(output_gradient, self.weights)