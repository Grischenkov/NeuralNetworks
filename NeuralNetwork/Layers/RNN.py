import numpy as np

from NeuralNetwork.Layers.Layer import Layer
from NeuralNetwork.Activations.Activation import Activation

class RNN(Layer):
    def __init__(self, depth: int, block_shape: int, activation: Activation) -> None:
        self.depth = depth
        self.output_shape = depth
        self.activation = activation
        self.block_shape = block_shape
        self.h_history = []
        self.x_history = []
    def compile(self, input_shape: int) -> None:
        self.input_shape = input_shape
        self.b_h = np.random.rand(self.block_shape, 1)
        self.b_y = np.random.rand(1, 1)
        self.w_xh = np.random.rand(self.block_shape, self.input_shape)
        self.w_hh = np.random.rand(self.block_shape, self.block_shape)
        self.w_hy = np.random.rand(1, self.block_shape)
        self.db_h = np.zeros(self.b_h.shape)
        self.db_y = np.zeros(self.b_y.shape)
        self.dw_xh = np.zeros(self.w_xh.shape)
        self.dw_hh = np.zeros(self.w_hh.shape)
        self.dw_hy = np.zeros(self.w_hy.shape)
    def forward(self, input: np.ndarray) -> np.ndarray:
        self.input = input[0]
        self.input = self.input.reshape((self.input.shape[0], self.input.shape[1], 1))
        h = np.zeros((self.w_hh.shape[0], 1))
        self.h_history = [h]
        self.x_history = []
        for i in range(self.depth):
            h = self.activation.function(np.matmul(self.w_xh, self.input[i]) + np.matmul(self.w_hh, h) + self.b_h)
            self.h_history.append(h)
            self.x_history.append(self.input[i])
        y = np.matmul(self.w_hy, h) + self.b_y
        return self.activation.function(y)
    def backward(self, output_gradient: np.ndarray, learning_rate: float) -> np.ndarray:
        self.db_h = np.zeros(self.b_h.shape)
        self.db_y = np.zeros(self.b_y.shape)
        self.dw_xh = np.zeros(self.w_xh.shape)
        self.dw_hh = np.zeros(self.w_hh.shape)
        self.dw_hy = np.zeros(self.w_hy.shape)
        self.dw_hy = np.matmul(output_gradient, self.h_history[-1].T)
        self.db_y = output_gradient
        dh = np.matmul(self.w_hy.T, output_gradient)
        for t in range(self.depth, 0, -1):
            temp = np.multiply(self.activation.derivative(self.h_history[t]), dh)
            self.db_h += temp
            self.dw_hh += np.matmul(temp, self.h_history[t - 1].T)
            self.dw_xh += np.matmul(temp, self.x_history[t - 1].T)
            dh = np.matmul(self.w_hh, temp)
        for d in [self.dw_xh, self.dw_hh, self.dw_hy, self.db_h, self.db_y]:
            np.clip(d, -1, 1, out=d)
        self.w_xh -= learning_rate * self.dw_xh
        self.w_hh -= learning_rate * self.dw_hh
        self.w_hy -= learning_rate * self.dw_hy
        self.b_h -= learning_rate * self.db_h
        self.b_y -= learning_rate * self.db_y