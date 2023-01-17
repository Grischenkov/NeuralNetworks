import numpy as np

from NeuralNetwork.Layers.Layer import Layer
from NeuralNetwork.Activations.Tanh import Tanh
from NeuralNetwork.Activations.Sigmoid import Sigmoid

class LSTM(Layer):
    def __init__(self, block_shape: int) -> None:
        self.output_shape = 1
        self.block_shape = block_shape
        self.c_history = []
        self.o_history = []
        self.g_history = []
        self.f_history = []
        self.i_history = []
        self.zo_history = []
        self.zf_history = []
        self.zi_history = []
        self.zg_history = []
    def compile(self, input_shape: int) -> None:
        self.input_shape = input_shape[2]
        self.depth = input_shape[1]
        self.w_xi = np.random.rand(self.block_shape, self.input_shape)
        self.w_xg = np.random.rand(self.block_shape, self.input_shape)
        self.b_i = np.random.rand(self.block_shape, 1)
        self.w_hg = np.random.rand(self.block_shape, self.block_shape)
        self.w_hi = np.random.rand(self.block_shape, self.block_shape)
        self.b_g = np.random.rand(self.block_shape, 1)
        self.w_xf = np.random.rand(self.block_shape, self.input_shape)
        self.b_f = np.random.rand(self.block_shape, 1)
        self.w_hf = np.random.rand(self.block_shape, self.block_shape)
        self.w_xo = np.random.rand(self.block_shape, self.input_shape)
        self.b_o = np.random.rand(self.block_shape, 1)
        self.w_ho = np.random.rand(self.block_shape, self.block_shape)
        self.w_y = np.random.rand(1, self.block_shape)
        self.b_y = np.random.rand(1, 1)
        self.dw_xi = np.zeros(self.w_xi.shape)
        self.dw_xg = np.zeros(self.w_xg.shape)
        self.db_i = np.zeros(self.b_i.shape)
        self.dw_hg = np.zeros(self.w_hg.shape)
        self.dw_hi = np.zeros(self.w_hi.shape)
        self.db_g = np.zeros(self.b_g.shape)
        self.dw_xf = np.zeros(self.w_xf.shape)
        self.db_f = np.zeros(self.b_f.shape)
        self.dw_hf = np.zeros(self.w_hf.shape)
        self.dw_xo = np.zeros(self.w_xo.shape)
        self.db_o = np.zeros(self.b_o.shape)
        self.dw_ho = np.zeros(self.w_ho.shape)
        self.dw_y = np.zeros(self.w_y.shape)
        self.db_y = np.zeros(self.b_y.shape)
    def forward(self, input: np.ndarray) -> np.ndarray:
        self.input = input[0]
        self.input = self.input.reshape((self.input.shape[0], self.input.shape[1], 1))
        h = np.zeros((self.w_xi.shape[0], 1))
        c = np.zeros((self.w_xi.shape[0], 1))
        self.h_history = [h]
        self.c_history = [c]
        self.x_history = []
        self.o_history = []
        self.g_history = []
        self.f_history = []
        self.i_history = []
        self.zo_history = []
        self.zf_history = []
        self.zi_history = []
        self.zg_history = []
        for t in range(self.depth):
            z_i = np.matmul(self.w_xi, self.input[t]) + np.matmul(self.w_hi, h) + self.b_i
            z_g = np.matmul(self.w_xg, self.input[t]) + np.matmul(self.w_hg, h) + self.b_g
            i = Sigmoid.function(z_i)
            g = Tanh.function(z_g)
            input_gate_out = np.multiply(i, g)
            z_f = np.matmul(self.w_xf, self.input[t]) + np.matmul(self.w_hf, h) + self.b_f
            f = Sigmoid.function(z_f)
            forget_gate_out = f
            z_o = np.matmul(self.w_xo, self.input[t]) + np.matmul(self.w_ho, h) + self.b_o
            o = Sigmoid.function(z_o)
            out_gate_out = o
            c = np.multiply(c, forget_gate_out) + input_gate_out
            h = np.multiply(out_gate_out, Tanh.function(c))
            self.x_history.append(self.input[t])
            self.h_history.append(h)
            self.c_history.append(c)
            self.o_history.append(o)
            self.g_history.append(g)
            self.f_history.append(f)
            self.i_history.append(i)
            self.zo_history.append(z_o)
            self.zf_history.append(z_f)
            self.zi_history.append(z_i)
            self.zg_history.append(z_g)
        y = np.matmul(self.w_y, h) + self.b_y
        return Sigmoid.function(y)
    def backward(self, output_gradient: np.ndarray, learning_rate: float) -> np.ndarray:
        self.dw_xi = np.zeros(self.w_xi.shape)
        self.dw_xg = np.zeros(self.w_xg.shape)
        self.db_i = np.zeros(self.b_i.shape)
        self.dw_hg = np.zeros(self.w_hg.shape)
        self.dw_hi = np.zeros(self.w_hi.shape)
        self.db_g = np.zeros(self.b_g.shape)
        self.dw_xf = np.zeros(self.w_xf.shape)
        self.db_f = np.zeros(self.b_f.shape)
        self.dw_hf = np.zeros(self.w_hf.shape)
        self.dw_xo = np.zeros(self.w_xo.shape)
        self.db_o = np.zeros(self.b_o.shape)
        self.dw_ho = np.zeros(self.w_ho.shape)
        self.dw_y = np.zeros(self.w_y.shape)
        self.db_y = np.zeros(self.b_y.shape)
        self.dw_y = np.matmul(output_gradient, self.h_history[-1].T)
        self.db_y = output_gradient
        dh = np.matmul(self.w_y.T, output_gradient)
        for t in range(self.depth - 1, -1, -1):
            db_o = dh * Tanh.function(self.c_history[t+1]) * Sigmoid.derivative(self.zo_history[t])
            self.db_o += db_o
            self.dw_xo += np.matmul(db_o, self.x_history[t].T)
            self.dw_ho += np.matmul(db_o, self.h_history[t].T)
            db_f = dh * self.o_history[t] * Tanh.derivative(self.c_history[t+1]) * self.c_history[t] * Sigmoid.derivative(self.zf_history[t])
            self.db_f += db_f
            self.dw_xf += np.matmul(db_f, self.x_history[t].T)
            self.dw_hf += np.matmul(db_f, self.h_history[t].T)
            db_i = dh * self.o_history[t] * Tanh.derivative(self.c_history[t+1]) * self.g_history[t] * Sigmoid.derivative(self.zi_history[t])
            self.db_i += db_i
            self.dw_xi += np.matmul(db_i, self.x_history[t].T)
            self.dw_hi += np.matmul(db_i, self.h_history[t].T)
            db_g = dh * self.o_history[t] * Tanh.derivative(self.c_history[t+1]) * self.i_history[t] * Tanh.derivative(self.zg_history[t])
            self.db_g += db_g
            self.dw_xg += np.matmul(db_g, self.x_history[t].T)
            self.dw_hg += np.matmul(db_g, self.h_history[t].T)
            dh = np.matmul(self.w_hf.T, self.f_history[t]) + np.matmul(self.w_hi.T, self.i_history[t]) + \
                np.matmul(self.w_ho.T, self.o_history[t]) + np.matmul(self.w_hg.T, self.g_history[t])
        self.w_xi -= learning_rate * self.dw_xi
        self.w_xg -= learning_rate * self.dw_xg
        self.b_i -= learning_rate * self.db_i
        self.w_hg -= learning_rate * self.dw_hg
        self.w_hi -= learning_rate * self.dw_hi
        self.b_g -= learning_rate * self.db_g
        self.w_xf -= learning_rate * self.dw_xf
        self.b_f -= learning_rate * self.db_f
        self.w_hf -= learning_rate * self.dw_hf
        self.w_xo -= learning_rate * self.dw_xo
        self.b_o -= learning_rate * self.db_o
        self.w_ho -= learning_rate * self.dw_ho
        self.w_y -= learning_rate * self.dw_y
        self.b_y -= learning_rate * self.db_y