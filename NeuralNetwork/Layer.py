import math
import NeuralNetwork.Activation
import numpy as np

class Layer:
    def __init__(self) -> None:
        self.input_shape = None
        self.output_shape = None
    def compile(self, input_shape: int or tuple) -> None:
        pass
    def forward(self, input: np.ndarray) -> np.ndarray:
        pass
    def backward(self, output_gradient: np.ndarray, learning_rate: float) -> np.ndarray:
        pass
class Input(Layer):
    def __init__(self, input_shape: int or tuple) -> None:
        self.compile(input_shape)
        self.output_shape = input_shape
    def compile(self, input_shape: int or tuple) -> None:
        self.input_shape = input_shape
    def forward(self, input: np.ndarray) -> np.ndarray:
        return input
    def backward(self, output_gradient: np.ndarray, learning_rate: float) -> np.ndarray:
        return output_gradient
class Dense(Layer):
    def __init__(self, shape: int, activation: NeuralNetwork.Activation) -> None:
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
class Flatten(Layer):
    def __init__(self) -> None:
        pass
    def compile(self, input_shape: tuple) -> None:
        self.input_shape = input_shape
        self.output_shape = self.input_shape[0] * self.input_shape[1] * self.input_shape[2]
    def forward(self, input: np.ndarray) -> np.ndarray:
        return np.reshape(input, (1, -1))
    def backward(self, output_gradient: np.ndarray, learning_rate: float) -> np.ndarray:
        return np.reshape(output_gradient, self.input_shape)