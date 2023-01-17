import numpy as np

from NeuralNetwork.Activations.Activation import Activation

class ReLU(Activation):
    def function(x):
        return np.maximum(x, 0)
    def derivative(x):
        return np.array(x >= 0).astype('int')