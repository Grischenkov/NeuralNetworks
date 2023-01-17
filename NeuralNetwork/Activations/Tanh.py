import numpy as np

from NeuralNetwork.Activations.Activation import Activation

class Tanh(Activation):
    def function(x):
        return np.tanh(x)
    def derivative(x):
        return 1-np.square(np.tanh(x))