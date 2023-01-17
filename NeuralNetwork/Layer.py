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
class MaxPool2D(Layer):
    def __init__(self, pool_size: tuple) -> None:
        self.pool_size = pool_size
    def compile(self, input_shape: tuple) -> None:
        self.input_shape = input_shape
        self.output_shape = (self.input_shape[0], math.ceil(self.input_shape[1]/self.pool_size[0]), math.ceil(self.input_shape[2]/self.pool_size[1]))
    def forward(self, input: np.ndarray) -> np.ndarray:
        self.input = input
        self.output = np.zeros(self.output_shape)
        for i in range(self.input.shape[1]):
            self.output[i] = self.__forward_pool(self.input[0][i])
        return np.array([self.output])
    def backward(self, output_gradient: np.ndarray, learning_rate: float) -> np.ndarray:
        gradient = np.zeros(self.input_shape)
        for i in range(output_gradient.shape[0]):
            gradient[i] = self.__backward_pool(gradient[i], output_gradient[i])
        return gradient
    def __forward_pool(self, input: np.ndarray) -> np.ndarray:
        result = []
        for i in range(0, input.shape[0], self.pool_size[0]):
            row = []
            for j in range(0, input.shape[1], self.pool_size[1]):
                slice = input[i:i+self.pool_size[0], j:j+self.pool_size[1]]
                row.append(slice[np.unravel_index(np.nanargmax(slice), slice.shape)])
            result.append(row)
        return np.array(result)
    def __backward_pool(self, gradient: np.ndarray, output_gradient: np.ndarray) -> np.ndarray:
        x = 0; y = 0
        for i in range(0, gradient.shape[0], self.pool_size[0]):
            for j in range(0, gradient.shape[1], self.pool_size[1]):
                slice = gradient[i:i+self.pool_size[0], j:j+self.pool_size[1]]
                slice[np.unravel_index(np.nanargmax(slice), slice.shape)] = output_gradient[y, x]
                gradient[i:i+self.pool_size[0], j:j+self.pool_size[1]] = slice
                x += 1
            x = 0; y += 1
        return gradient