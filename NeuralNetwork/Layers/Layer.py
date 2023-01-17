import numpy as np

class Layer:
    def __init__(self) -> None:
        self.input_shape = None
        self.output_shape = None
    def compile(self, input_shape: int or tuple) -> None:
        pass
    def forward(self, input: np.ndarray) -> np.ndarray:
        pass
    def backward(self, output_gradient: np.ndarray, learning_rate: float) -> np.ndarray:
        pass