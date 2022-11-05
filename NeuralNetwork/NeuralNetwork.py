import numpy as np

class Network:
    def __init__(self):
        self.history = []
        self.layers = []
        self.loss = None
    def add (self, layer):
        self.layers.append(layer)
    def compile(self, loss):
        self.loss = loss
        for i in range(1, len(self.layers)):
            self.layers[i].compile(self.layers[i-1].size)
    def fit(self, X_train, Y_train, epochs, learning_rate):
        for i in range(epochs):
            loss = 0
            for j in range(len(X_train)):
                output = self.__forward(X_train[j])
                loss += self.loss.function(Y_train[j], output)
                error = self.loss.derivative(Y_train[j], output)
                self.__backward(error, learning_rate)
            loss /= len(X_train)
            self.history.append(loss)
            print(f'Epoch {i}/{epochs}: loss={loss}')
    def predict(self, X):
        result = []
        for i in range(len(X)):
            output = self.__forward(X[i])
            result.append(output[0])
        return result
    def __forward(self, input):
        output = np.array([input])
        for layer in self.layers:
            output = layer.forward(output)
        return output
    def __backward(self, error, learning_rate):
        gradient = error
        for layer in reversed(self.layers):
            gradient = layer.backward(gradient, learning_rate)