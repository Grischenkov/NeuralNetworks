import numpy as np

class Loss:
    def function(x):
        pass
    def derivative(x):
        pass
class MSE(Loss):
    def function(y_true, y_pred):
        return np.mean(np.power(y_true-y_pred, 2))
    def derivative(y_true, y_pred):
        return 2*(y_pred-y_true)/y_true.size