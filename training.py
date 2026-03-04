import numpy as np
from engine import NNEngine


x = np.array([[2.0, 3.0, -1.0], [3.0, -1.0, 0.5], [0.5, 1.0, 1.0], [1.0, 1.0, -1.0]])
x_normalized = (x - np.mean(x)) / np.std(x)
y = np.array([1.0, -1.0, -1.0, 1.0])
#y = np.array([10, 20, 30, 40])
input_data = np.hstack((x_normalized, y.reshape(-1, 1)))

n_layers = 2
hidden_layers_dim = [10]  # number of neurons in each hidden layer. len = n_layers - 1
activation_funcs = ['tanh', None]

np.random.seed(100)


engine = NNEngine(input_data, n_layers, hidden_layers_dim, activation_funcs, classification=False)
engine.init_params()


for k in range(10000):

    # forward
    cost, perf, y, y_pred = engine.forward_propagation()

    # backward
    engine.zero_grad()
    engine.backward_propagation(y, y_pred)

    # update weights
    # learning_rate = 1.0 - 0.9 * (k + 1) / 100
    learning_rate = 0.0001
    engine.gradient_descent(learning_rate)

    # print(f"step {k}. loss={cost}; perf={cost}; y_pred={y_pred.ravel()}")
    print(f"step {k}. loss={cost}; y_pred={y_pred.ravel()}")
