import math
import numpy as np

from NeuralNetwork.Layers.Layer import Layer

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
            self.output[i] = self.__forward_pool(self.pool_size, self.input[0][i])
        return np.array([self.output])
    def backward(self, output_gradient: np.ndarray, learning_rate: float) -> np.ndarray:
        gradient = np.zeros(self.input_shape)
        for i in range(output_gradient.shape[0]):
            gradient[i] = self.__backward_pool(self.pool_size, self.input[0][i], gradient[i], output_gradient[i])
        return gradient
    def __forward_pool(self, pool_size : tuple, input: np.ndarray) -> np.ndarray:
        result = []
        for i in range(0, input.shape[0], pool_size[0]):
            row = []
            for j in range(0, input.shape[1], pool_size[1]):
                slice = input[i:i+pool_size[0], j:j+pool_size[1]]
                row.append(slice[np.unravel_index(np.nanargmax(slice), slice.shape)])
            result.append(row)
        return np.array(result)
    def __backward_pool(self, pool_size : tuple, input: np.ndarray, gradient: np.ndarray, output_gradient: np.ndarray) -> np.ndarray:
        x = 0; y = 0
        for i in range(0, gradient.shape[0], pool_size[0]):
            for j in range(0, gradient.shape[1], pool_size[1]):
                input_slice = input[i:i+pool_size[0], j:j+pool_size[1]]
                gradient_slice = gradient[i:i+pool_size[0], j:j+pool_size[1]]
                gradient_slice[np.unravel_index(np.nanargmax(input_slice), input_slice.shape)] = output_gradient[y, x]
                gradient[i:i+pool_size[0], j:j+pool_size[1]] = gradient_slice
                x += 1
            x = 0; y += 1
        return gradient