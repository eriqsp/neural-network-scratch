import numpy as np


class Activations:
    def __init__(self):
        pass

    """ activation functions below """
    def activation_function(self, v, name=None):
        if name == 'relu':
            return self._relu(v)
        if name == 'softmax':
            return self._softmax(v)
        if name == 'tanh':
            return self._tanh(v)
        return v

    @staticmethod
    def _relu(v):
        return np.maximum(0, v)

    @staticmethod
    def _softmax(v):
        exps = np.exp(v - np.max(v, axis=1, keepdims=True))
        return exps / np.sum(exps, axis=1, keepdims=True)

    @staticmethod
    def _tanh(v):
        return np.tanh(v)

    @staticmethod
    def deriv_activation(v, name='relu'):
        if name == 'tanh':
            return 1 - np.tanh(v) ** 2
        return v > 0
