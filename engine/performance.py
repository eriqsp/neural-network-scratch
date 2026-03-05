import numpy as np


class Performance:
    def __init__(self):
        pass

    """ cost functions below """
    # mean-squared error
    @staticmethod
    def mse_cost(y, y_p):
        n = y.size
        return sum((y - y_p.ravel()) ** 2) * (1 / n) if n > 0 else None

    # categorical cross-entropy
    def cce_cost(self, y, y_p):
        return -sum(sum(y * np.log(np.clip(y_p, 1e-15, 1.0))))

    """ performance evaluation functions """
    @staticmethod
    def accuracy(y, y_p):  # for classification problems
        y_p = np.argmax(y_p, 1).astype(str)
        return np.sum(y_p == y) / y.size

    @staticmethod
    def r_squared(y, y_p):  # for regression problems
        corr_matrix = np.corrcoef(y, y_p.ravel())
        r = corr_matrix[0, 1]
        r2 = r ** 2
        return r2
