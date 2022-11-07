import numpy as np

class Loss:
    def function(x):
        pass
    def derivative(x):
        pass
class MSE(Loss):
    def function(y_true, y_pred):
        return np.square(np.subtract(y_pred, y_true)).mean()
    def derivative(y_true, y_pred):
        return 2 * np.subtract(y_pred, y_true) / y_true.size