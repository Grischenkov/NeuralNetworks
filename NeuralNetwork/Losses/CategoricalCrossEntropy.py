import numpy as np

from NeuralNetwork.Losses.Loss import Loss

class CategoricalCrossEntropy(Loss):
    def function(y_true, y_pred):
        return -np.sum(y_true * np.log(y_pred + 10**-100))
    def derivative(y_true, y_pred):
        return -y_true/(y_pred + 10**-100)