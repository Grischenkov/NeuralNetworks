import numpy as np

from scipy import signal

from NeuralNetwork.Layers.Layer import Layer
from NeuralNetwork.Activations.Activation import Activation

class Conv2D(Layer):
    def __init__(self, filters: int, kernel_size: tuple, activation: Activation) -> None:
        self.filters = filters
        self.activation = activation
        self.kernel_size = kernel_size
    def compile(self, input_shape: tuple) -> None:
        self.input_shape = input_shape
        self.kernels_shape = (self.filters, self.input_shape[0], self.kernel_size[0], self.kernel_size[1])
        self.output_shape = (self.filters, self.input_shape[1]-self.kernel_size[0]+1, self.input_shape[2]-self.kernel_size[1]+1)
        self.kernels = np.random.randn(*self.kernels_shape)
        self.biases = np.random.randn(*self.output_shape)
    def forward(self, input: np.ndarray) -> np.ndarray:
        self.input = input
        self.activation_input = np.copy(self.biases)
        for i in range(self.filters):
            for j in range(self.input.shape[1]):
                self.activation_input[i] += signal.correlate2d(self.input[0][j], self.kernels[i, j], "valid")
        return np.array([self.activation.function(self.activation_input)])
    def backward(self, output_gradient: np.ndarray, learning_rate: float) -> np.ndarray:
        kernels_gradient = np.zeros(self.kernels_shape)
        input_gradient = np.zeros(self.input_shape)
        for i in range(self.filters):
            for j in range(self.input.shape[1]):
                kernels_gradient[i, j] = signal.correlate2d(self.input[0][j], output_gradient[i], "valid")
                input_gradient[j] += signal.convolve2d(output_gradient[i], self.kernels[i, j], "full")
        self.kernels -= learning_rate * kernels_gradient
        self.biases -= learning_rate * output_gradient
        return input_gradient