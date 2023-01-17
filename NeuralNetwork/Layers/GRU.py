import numpy as np

from NeuralNetwork.Layers.Layer import Layer
from NeuralNetwork.Activations.Tanh import Tanh
from NeuralNetwork.Activations.Sigmoid import Sigmoid

class GRU(Layer):
    def __init__(self, block_shape: int) -> None:
        self.output_shape = 1
        self.block_shape = block_shape
        self.h_history = []
        self.x_history = []
        self.z_history = []
        self.r_history = []
        self.temp_history = []
        self.y = 0
    def compile(self, input_shape: int) -> None:
        self.input_shape = input_shape[2]
        self.depth = input_shape[1]
        self.w_z = np.random.rand(self.block_shape, self.input_shape)
        self.u_z = np.random.rand(self.block_shape, self.block_shape)
        self.w_r = np.random.rand(self.block_shape, self.input_shape)
        self.u_r = np.random.rand(self.block_shape, self.block_shape)
        self.w_h = np.random.rand(self.block_shape, self.input_shape)
        self.u_h = np.random.rand(self.block_shape, self.block_shape)
        self.w_y = np.random.rand(1, self.block_shape)
        self.b_y = np.random.rand(1, 1)
        self.dw_z = np.zeros(self.w_z.shape)
        self.du_z = np.zeros(self.u_z.shape)
        self.dw_r = np.zeros(self.w_r.shape)
        self.du_r = np.zeros(self.u_r.shape)
        self.dw_h = np.zeros(self.w_h.shape)
        self.du_h = np.zeros(self.u_h.shape)
        self.dw_y = np.zeros(self.w_y.shape)
        self.db_y = np.zeros(self.b_y.shape)
    def forward(self, input: np.ndarray) -> np.ndarray:
        self.input = input[0]
        self.input = self.input.reshape((self.input.shape[0], self.input.shape[1], 1))
        h = np.zeros((self.w_h.shape[0], 1))
        self.h_history = [h]
        self.x_history = []
        self.z_history = []
        self.r_history = []
        for i in range(self.depth):
            z = Sigmoid.function(np.matmul(self.w_z, self.input[i]) + np.matmul(self.u_z, h))
            r = Sigmoid.function(np.matmul(self.w_r, self.input[i]) + np.matmul(self.u_r, h))
            temp = Tanh.function(np.matmul(self.w_h, self.input[i]) + np.matmul(self.u_h, np.multiply(r, h)))
            h = np.multiply(z, h) + np.multiply((1 - z), temp)
            self.x_history.append(self.input[i])
            self.z_history.append(z)
            self.r_history.append(r)
            self.h_history.append(h)
            self.temp_history.append(temp)
        self.y = np.matmul(self.w_y, h) + self.b_y
        return Sigmoid.function(self.y)
    def backward(self, output_gradient: np.ndarray, learning_rate: float) -> np.ndarray:
        self.dw_z = np.zeros(self.w_z.shape)
        self.du_z = np.zeros(self.u_z.shape)
        self.dw_r = np.zeros(self.w_r.shape)
        self.du_r = np.zeros(self.u_r.shape)
        self.dw_h = np.zeros(self.w_h.shape)
        self.du_h = np.zeros(self.u_h.shape)
        self.dw_y = np.zeros(self.w_y.shape)
        self.db_y = np.zeros(self.b_y.shape)
        self.dw_y = np.matmul(output_gradient, self.h_history[-1].T)
        self.db_y = output_gradient
        dh = np.matmul(self.w_y.T, output_gradient)
        for t in range(self.depth - 1, -1, -1):
            d1 = np.multiply(self.z_history[t], dh)
            d2 = np.multiply(self.h_history[t], dh)
            d3 = np.multiply(self.temp_history[t], dh)
            d4 = d3 * -1
            d5 = d2 + d4
            d6 = np.multiply(1-self.z_history[t], dh)
            d7 = np.multiply(d5, Sigmoid.derivative(self.z_history[t]))
            d8 = np.multiply(d6, Tanh.derivative(self.temp_history[t]))
            d9 = np.matmul(self.w_h.T, d8)
            d10 = np.matmul(self.u_h.T, d8)
            d11 = np.matmul(self.w_z.T, d7)
            d12 = np.matmul(self.u_z.T, d7)
            d14 = np.multiply(d10, self.r_history[t])
            d15 = np.multiply(d10, self.h_history[t])
            d16 = np.multiply(d15, Sigmoid.derivative(self.r_history[t]))
            d17 = np.matmul(self.u_r.T, d7)
            d13 = np.matmul(self.w_r.T, d7)
            dx = d9 + d11 + d13
            dh = d12 + d14 + d1 + d17
            self.dw_r += np.matmul(d16, self.x_history[t].T)
            self.dw_z += np.matmul(d7, self.x_history[t].T)
            self.dw_h += np.matmul(d8, self.x_history[t].T)
            self.du_r += np.matmul(d16, self.h_history[t].T)
            self.du_z += np.matmul(d7, self.h_history[t].T)
            self.du_h += np.matmul(d8, np.multiply(self.h_history[t], self.r_history[t]).T)
        for d in [self.dw_r, self.dw_z, self.dw_h, self.du_r, self.du_z, self.du_h, self.dw_y, self.db_y]:
            np.clip(d, -1, 1, out=d)
        self.w_z -= learning_rate * self.dw_z
        self.w_r -= learning_rate * self.dw_r
        self.w_h -= learning_rate * self.dw_h
        self.w_y -= learning_rate * self.dw_y
        self.b_y -= learning_rate * self.db_y
        self.u_z -= learning_rate * self.du_z
        self.u_r -= learning_rate * self.du_r
        self.u_h -= learning_rate * self.du_h