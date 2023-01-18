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
            result = []
            for z in range(0, self.input[0][i].shape[0], self.pool_size[0]):
                row = []
                for j in range(0, self.input[0][i].shape[1], self.pool_size[1]):
                    slice = self.input[0][i][z:z+self.pool_size[0], j:j+self.pool_size[1]]
                    row.append(slice[np.unravel_index(np.nanargmax(slice), slice.shape)])
                result.append(row)
            self.output[i] = np.array(result)
        return np.array([self.output])
    def backward(self, output_gradient: np.ndarray, learning_rate: float) -> np.ndarray:
        gradient = np.zeros(self.input_shape)
        for i in range(output_gradient.shape[0]):
            x = 0; y = 0
            for z in range(0, gradient[i].shape[0], self.pool_size[0]):
                for j in range(0, gradient[i].shape[1], self.pool_size[1]):
                    input_slice = self.input[0][i][z:z+self.pool_size[0], j:j+self.pool_size[1]]
                    gradient_slice = gradient[i][z:z+self.pool_size[0], j:j+self.pool_size[1]]
                    gradient_slice[np.unravel_index(np.nanargmax(input_slice), input_slice.shape)] = output_gradient[i][y, x]
                    gradient[i][z:z+self.pool_size[0], j:j+self.pool_size[1]] = gradient_slice
                    x += 1
                x = 0; y += 1
        return gradient