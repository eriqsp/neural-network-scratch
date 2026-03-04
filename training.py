import numpy as np
from engine import NNEngine


x = np.array([[2.0, 3.0, -1.0], [3.0, -1.0, 0.5], [0.5, 1.0, 1.0], [1.0, 1.0, -1.0]])
y = np.array([1.0, -1.0, -1.0, 1.0])
input_data = np.hstack((x, y.reshape(-1, 1)))

n_layers = 2
hidden_layers_dim = [10]
activation_funcs = ['relu', None]

np.random.seed(100)


engine = NNEngine(input_data, n_layers, hidden_layers_dim, activation_funcs, classification=False)
engine.init_params()


for k in range(100):

    # forward
    cost, perf, y, y_pred = engine.forward_propagation()

    # backward
    engine.zero_grad()
    engine.backward_propagation(y, y_pred)  # TODO: i think the problem is here

    # update weights
    engine.gradient_descent(0.01)

    # print(f"step {k}. loss={cost}; perf={cost}; y_pred={y_pred.ravel()}")
    print(f"step {k}. loss={cost}; y_pred={y_pred.ravel()}")
