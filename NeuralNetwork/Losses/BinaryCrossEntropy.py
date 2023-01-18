import numpy as np

from NeuralNetwork.Losses.Loss import Loss

class BinaryCrossEntropy(Loss):
    def function(y_true, y_pred):
        return np.mean(-y_true * np.log(y_pred) - (1 - y_true) * np.log(1 - y_pred))
    def derivative(y_true, y_pred):
        return ((1 - y_true) / (1 - y_pred) - y_true / y_pred) / np.size(y_true)