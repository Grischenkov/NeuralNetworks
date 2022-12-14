import numpy as np

class Activation:
    def function(x):
        pass
    def derivative(x):
        pass
class Tanh(Activation):
    def function(x):
        return np.tanh(x)
    def derivative(x):
        return 1-np.square(np.tanh(x))
class Sigmoid(Activation):
    def function(x):
        return 1 / (1 + np.exp(-x))
    def derivative(x):
        return np.exp(-x) / np.square(1 + np.exp(-x))
class ReLU(Activation):
    def function(x):
        return np.maximum(x, 0)
    def derivative(x):
        return np.array(x >= 0).astype('int')