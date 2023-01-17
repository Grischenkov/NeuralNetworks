import numpy as np

from NeuralNetwork.Layers.Layer import Layer

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