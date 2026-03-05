import numpy as np


class NNEngine:
    def __init__(self, layers, classification=True):
        self.classification = classification

        self.n_layers = len(layers)
        self.layers = layers  # list of tuples - [(n_units, activation_function)]

        self.activations_values = None  # stores the activation values for each unit

        self.weights = None
        self.bias = None

        self.dW = None  # derivatives for the weights
        self.db = None  # derivatives for the biases

        self.Z = None  # Z = WX + b - input for the activation function

    def init_params(self, x):
        n_units = self.layers[0][0]
        shape = (n_units, x.shape[1])
        self.weights = [np.random.normal(0, self._weight_init(self.layers[0][1], shape), size=shape)]
        self.bias = [np.zeros((n_units, 1))]

        for n in range(1, self.n_layers - 1):
            n_units = self.layers[n][0]
            shape = (n_units, self.layers[n - 1][0])
            self.weights.append(np.random.normal(0, self._weight_init(self.layers[n][1], shape), size=shape))
            self.bias.append(np.zeros((n_units, 1)))

        # last layer
        n_units = self.layers[self.n_layers - 2][0]
        n_units_last = self.layers[self.n_layers - 1][0]
        shape = (n_units_last, n_units)
        self.weights.append(np.random.normal(0, self._weight_init(self.layers[self.n_layers - 1][1], shape), size=shape))
        self.bias.append(np.zeros((n_units_last, 1)))

        self.dW = [np.zeros_like(array) for array in self.weights]
        self.db = [np.zeros_like(array) for array in self.bias]

        self.Z = [np.zeros_like(array) for array in self.bias]

    @staticmethod
    def _weight_init(activation, shape):
        if activation == 'relu':
            return np.sqrt(2 / shape[1])  # uses He initialization (specific for ReLU activation)
        return np.sqrt(2 / (shape[0] + shape[1]))  # uses Xavier initialization as default

    def forward_propagation(self, x, y, batch_size=None):
        if batch_size is None:
            xb, yb = x, y
        else:
            ri = np.random.permutation(x.shape[0])[:batch_size]  # get only a fraction of the total dataset. Reduces computational cost (SGD)
            xb, yb = x[ri], y[ri]

        q = xb
        self.activations_values = [q]
        for n in range(0, self.n_layers):
            self.Z[n] = q @ self.weights[n].T + self.bias[n].T
            if n != self.n_layers - 1:
                q = self.activation_function(self.Z[n], self.layers[n][1])
                self.activations_values.append(q)  # storing activations to use on backward propagation

        # final layer transformation
        y_pred = self.activation_function(self.Z[self.n_layers - 1], self.layers[self.n_layers - 1][1])

        if self.classification:
            perf = self.accuracy(yb, y_pred)
            cost = self.cce_cost(yb, y_pred)
        else:
            cost = self.mse_cost(yb, y_pred)
            perf = self.r_squared(yb, y_pred)

        return cost, perf, yb, y_pred

    def backward_propagation(self, y, y_pred):
        n = y.size  # mini-batch size

        if self.classification:
            dZ = y_pred - self._y_transform(y)  # derivative of the cost function w.r.t. z
        else:
            dZ = 2 * (y_pred - y.reshape(-1, 1))

        for l in range(self.n_layers - 1, -1, -1):
            # chain rule makes it easier to compute the derivatives
            a_prev = self.activations_values[l]

            # calculate gradients for weights and bias
            self.dW[l] = (1 / n) * (dZ.T @ a_prev)
            self.db[l] = (1 / n) * np.sum(dZ, axis=0, keepdims=True).T

            if l > 0:
                dZ = (dZ @ self.weights[l]) * self.derivActivation(self.Z[l - 1], name=self.layers[l - 1][1])

    # for classification: transform the y output into a matrix of entries (a, M) where M is the number os classes
    @staticmethod
    def _y_transform(y):
        classes = np.unique(y)
        one_hot = (y[:, np.newaxis] == classes)
        return one_hot.astype('float64')

    # update weights with gradient descent
    def gradient_descent(self, learning_rate: float):
        for layer in range(self.n_layers):
            self.weights[layer] -= learning_rate * self.dW[layer]
            self.bias[layer] -= learning_rate * self.db[layer]


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
    def derivActivation(Z, name='relu'):
        if name == 'tanh':
            return 1 - np.tanh(Z) ** 2
        return Z > 0


    """ cost functions below """
    # mean-squared error
    @staticmethod
    def mse_cost(y, y_p):
        n = y.size
        return sum((y - y_p.ravel()) ** 2) * (1 / n) if n > 0 else None

    # categorical cross-entropy
    def cce_cost(self, y, y_p):
        return -sum(sum(self._y_transform(y) * np.log(np.clip(y_p, 1e-15, 1.0))))


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
