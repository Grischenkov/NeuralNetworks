import numpy as np

from NeuralNetwork.Activations.Activation import Activation

class Sigmoid(Activation):
    def function(x):
        return 1 / (1 + np.exp(-x))
    def derivative(x):
        return np.exp(-x) / np.square(1 + np.exp(-x))