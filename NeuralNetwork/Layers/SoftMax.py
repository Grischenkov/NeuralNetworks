import numpy as np

from NeuralNetwork.Layers.Layer import Layer

class SoftMax(Layer):
    def __init__(self) -> None:
        pass
    def compile(self, input_shape: tuple) -> None:
        self.input_shape = input_shape
        self.output_shape = input_shape
    def forward(self, input: np.ndarray) -> np.ndarray:
        self.input = input[0]
        tmp = np.exp(self.input)
        self.output = tmp / np.sum(tmp)
        return np.array(self.output)
    def backward(self, output_gradient: np.ndarray, learning_rate: float) -> np.ndarray:
        n = np.size(self.output)
        return np.array([np.dot((np.identity(n) - self.output.T) * self.output, output_gradient)])