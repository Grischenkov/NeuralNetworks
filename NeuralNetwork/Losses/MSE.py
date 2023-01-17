import numpy as np

from NeuralNetwork.Losses.Loss import Loss

class MSE(Loss):
    def function(y_true, y_pred):
        return np.square(np.subtract(y_pred, y_true)).mean()
    def derivative(y_true, y_pred):
        return 2 * np.subtract(y_pred, y_true) / y_true.size