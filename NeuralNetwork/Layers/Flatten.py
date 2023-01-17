import numpy as np

from NeuralNetwork.Layers.Layer import Layer

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