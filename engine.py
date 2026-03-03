import numpy as np


class Engine:
    def __init__(self, classification=True):
        self.classification = classification
        self.input_x = None  # dimensions = (a,b), where a=number of examples and b=number of features
        self.y = None  # dimensions = a - output vector
        self.weights = None  # dimensions = (a,b,c), where c=number of layers
        self.bias = None  # dimensions = (a,c)
        self.n_layers = None

    def forward_propagation(self):
        q = self.input_x
        z = np.matmul(q, self.weights[0]) + self.bias[0]  # transpose weights and bias before this operation
        for n in range(1, self.n_layers - 1):
            q = self.relu(z)
            z = np.matmul(q, self.weights[n]) + self.bias[n]

        s = self.softmax(z)

        acc = 0
        if self.classification:
            self.y_transform()
            # acc = self.accuracy(self.y, s)

        cost = self.cce_cost(self.y, s)

        return cost, acc

    # For classification: transform the y output into a matrix of entries (a, M) where M is the number os classes
    def y_transform(self):
        structure = np.zeros((self.y.size, self.y.max + 1))
        structure[np.arange(self.y.size), self.y] = 1
        self.y = structure

    # categorical cross-entropy
    @staticmethod
    def cce_cost(y, y_p):
        return -sum(sum(y * np.log(y_p)))

    # mean-squared error
    @staticmethod
    def mse_cost(y, y_p):
        n = len(y)
        return sum((y - y_p) ** 2) * 1 / n if n > 0 else None

    @staticmethod
    def relu(v):
        return np.maximum(0, v)

    @staticmethod
    def softmax(v):
        return np.exp(v) / sum(np.exp(v))

    @staticmethod
    def accuracy(y, y_p):
        predicted = y_p.map(np.argmax)  # not exactly it but yeah

