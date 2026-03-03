import numpy as np


# TODO: create loop for the training procedure (for each mini-batch)
# TODO: implement derivative for other types of activation functions


class Engine:
    def __init__(self, input_data, n_layers, hidden_layers_dim, classification=True):
        self.classification = classification
        self.n_layers = n_layers
        self.hidden_layers_dim = hidden_layers_dim

        self.x = input_data[:, :-1]  # dimensions = (a,b), where a=number of examples and b=number of features
        self.y = input_data[:, -1]  # dimensions = (a,) - output vector

        self.weights = None  # dimensions = (a,b,c), where c=number of layers
        self.bias = None  # dimensions = (a,c)

        self.dW = None
        self.db = None

        self.Z = None  # Z = WX + b - input for the activation function

    def init_params(self):
        self.weights = np.array([np.random.uniform(-1, 1, size=(self.x[1].size, self.hidden_layers_dim[0]))])
        self.bias = np.array([np.random.uniform(-1, 1, size=(1, self.hidden_layers_dim[0]))])

        for layer in range(1, self.n_layers - 1):
            self.weights[layer] = np.random.uniform(-1, 1, size=(self.hidden_layers_dim[layer - 1], self.hidden_layers_dim[layer]))
            self.bias[layer] = np.random.uniform(-1, 1, size=(1, self.hidden_layers_dim[layer]))

        # last layer
        self.weights[self.n_layers - 1] = np.random.uniform(-1, 1, size=(self.hidden_layers_dim[self.n_layers - 2], 1))
        self.bias[self.n_layers - 1] = np.random.uniform(-1, 1, size=1)

        self.dW = np.zeros_like(self.weights)
        self.db = np.zeros_like(self.bias)

        self.Z = np.zeros_like(self.bias)

    def forward_propagation(self, activation_funcs):
        q = self.x
        self.Z[0] = q.dot(self.weights[0]) + self.bias[0]  # transpose weights and bias before this operation
        for n in range(1, self.n_layers):
            q = self.activation_function(self.Z[n - 1], activation_funcs[0])
            self.Z[n] = q.dot(self.weights[n]) + self.bias[n]

        # final layer transformation
        y_pred = self.activation_function(self.Z[self.n_layers - 1], activation_funcs[1])

        if self.classification:
            perf = self.accuracy(self.y, y_pred)
            cost = self.cce_cost(self.y, y_pred)
        else:
            cost = self.mse_cost(self.y, y_pred)
            perf = cost

        return cost, perf, y_pred

    def backward_propagation(self, y_pred, activation_funcs):
        n = self.y.size
        dZ = y_pred - self.y_transform()  # derivative of the cost function w.r.t. z
        for l in range(self.n_layers - 1, -1, -1):
            # chain rule makes it easier to compute the derivatives
            dq = dZ.dot(self.weights[l])

            self.dW[l] = (1 / n) * dZ.T.dot(dq)
            self.db[l] = (1 / n) * dZ[1].sum()

            dZ = dq * self.derivActivation(self.Z[l], name=activation_funcs[0])

    # update weights with gradient descent
    def gradient_descent(self, learning_rate):
        self.weights += -learning_rate * self.dW
        self.bias += -learning_rate * self.db

    def derivActivation(self, Z, name='relu'):
        return Z > 0

    # for classification: transform the y output into a matrix of entries (a, M) where M is the number os classes
    def y_transform(self):
        y_matrix = np.zeros((self.y.size, self.y.max + 1))
        y_matrix[np.arange(self.y.size), self.y] = 1
        return y_matrix


    """ cost functions below """
    # mean-squared error
    @staticmethod
    def mse_cost(y, y_p):
        n = y.size
        return sum((y - y_p) ** 2) * (1 / n) if n > 0 else None

    # categorical cross-entropy
    @staticmethod
    def cce_cost(y, y_p):
        return -sum(sum(y * np.log(y_p)))


    """ activation functions below """
    def activation_function(self, v, name='relu'):
        if name == 'softmax':
            return self._softmax(v)
        return self._relu(v)

    @staticmethod
    def _relu(v):
        return np.maximum(0, v)

    @staticmethod
    def _softmax(v):
        return np.exp(v) / sum(np.exp(v))


    """ performance evaluation functions """
    @staticmethod
    def accuracy(y, y_p):  # for classification problems
        y_p = np.argmax(y_p, 0)
        return np.sum(y_p == y) / y.size
