import numpy as np


# TODO: implement derivative for other types of activation functions


class NNEngine:
    def __init__(self, input_data, n_layers, hidden_layers_dim, activations, classification=True):
        self.classification = classification
        self.n_layers = n_layers
        self.hidden_layers_dim = hidden_layers_dim

        assert len(self.hidden_layers_dim) == self.n_layers - 1, "The length of the parameter hidden_layers_dim must match the number of hidden layers"

        self.activations = activations

        self.x = input_data[:, :-1]  # dimensions = (a,b), where a=number of examples and b=number of features
        self.y = input_data[:, -1]  # dimensions = (a,) - output vector

        self.weights = None  # dimensions = (a,b,c), where c=number of layers
        self.bias = None  # dimensions = (a,c)

        self.dW = None
        self.db = None

        self.Z = []  # Z = WX + b - input for the activation function

    def init_params(self, classes=None):
        self.weights = [np.random.uniform(-1, 1, size=(self.hidden_layers_dim[0], self.x.shape[1]))]
        self.bias = [np.random.uniform(-1, 1, size=(self.hidden_layers_dim[0], 1))]

        for layer in range(1, self.n_layers - 1):
            self.weights.append(np.random.uniform(-1, 1, size=(self.hidden_layers_dim[layer], self.hidden_layers_dim[layer - 1])))
            self.bias.append(np.random.uniform(-1, 1, size=(self.hidden_layers_dim[layer], 1)))

        # last layer
        self.weights.append(np.random.uniform(-1, 1, size=(1 if classes is None else classes, self.hidden_layers_dim[self.n_layers - 2])))
        self.bias.append(np.random.uniform(-1, 1, size=(1 if classes is None else classes, 1)))

        self.zero_grad()

    def forward_propagation(self, batch_size=None):
        if batch_size is None:
            xb, yb = self.x, self.y
        else:
            ri = np.random.permutation(self.x.shape[0])[:batch_size]  # get only a fraction of the total dataset. Reduces computational cost (SGD)
            xb, yb = self.x[ri], self.y[ri]

        q = xb
        for n in range(0, self.n_layers):
            self.Z.append(q.dot(self.weights[n].T) + self.bias[n].T)  # TODO: transpose weights and bias before this operation
            if n != self.n_layers - 1:
                q = self.activation_function(self.Z[n], self.activations[0])

        # final layer transformation
        y_pred = self.activation_function(self.Z[self.n_layers - 1], self.activations[1 if len(self.activations) > 1 else 0])

        if self.classification:
            perf = self.accuracy(yb, y_pred)
            cost = self.cce_cost(yb, y_pred)
        else:
            cost = self.mse_cost(yb, y_pred)
            perf = cost

        return cost, perf, yb, y_pred

    def backward_propagation(self, y, y_pred):
        n = self.y.size  # mini-batch size

        if self.classification:
            dZ = y_pred - self._y_transform(y)  # derivative of the cost function w.r.t. z
        else:
            dZ = y_pred - y.reshape(-1, 1)  # for regression is the same logic as for classification, but without the classes

        dq = 1
        for l in range(self.n_layers - 1, -1, -1):
            # chain rule makes it easier to compute the derivatives
            if l != self.n_layers - 1:
                dZ = dq * self.derivActivation(self.Z[l], name=self.activations[0])
            dq = dZ.dot(self.weights[l])

            self.dW[l] = (1 / n) * dZ.T.dot(dq)
            self.db[l] = (1 / n) * dZ.sum(axis=0).T.reshape(-1, 1)

    # for classification: transform the y output into a matrix of entries (a, M) where M is the number os classes
    @staticmethod
    def _y_transform(y):
        y_matrix = np.zeros((y.size, y.max() + 1))
        y_matrix[np.arange(y.size), y] = 1
        return y_matrix

    # update weights with gradient descent
    def gradient_descent(self, learning_rate: float):
        for layer in range(self.n_layers):
            self.weights[layer] += -learning_rate * self.dW[layer]
            self.bias[layer] += -learning_rate * self.db[layer]

    def zero_grad(self):
        self.dW = [np.zeros_like(array) for array in self.weights]
        self.db = [np.zeros_like(array) for array in self.bias]


    """ activation functions below """
    def activation_function(self, v, name=None):
        if name == 'relu':
            return self._relu(v)
        if name == 'softmax':
            return self._softmax(v)
        return v

    @staticmethod
    def _relu(v):
        return np.maximum(0, v)

    @staticmethod
    def _softmax(v):
        return np.exp(v) / sum(np.exp(v))

    def derivActivation(self, Z, name='relu'):
        return Z > 0


    """ cost functions below """
    # mean-squared error
    @staticmethod
    def mse_cost(y, y_p):
        n = y.size
        return sum((y - y_p.ravel()) ** 2) * (1 / n) if n > 0 else None

    # categorical cross-entropy
    @staticmethod
    def cce_cost(y, y_p):
        return -sum(sum(y * np.log(y_p)))


    """ performance evaluation functions """
    @staticmethod
    def accuracy(y, y_p):  # for classification problems
        y_p = np.argmax(y_p, 0)
        return np.sum(y_p == y) / y.size
