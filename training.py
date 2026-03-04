import numpy as np
from engine import NNEngine
from sklearn.datasets import fetch_openml

# mnist = fetch_openml('mnist_784', version=1, as_frame=False)
#
# x, y = mnist["data"], mnist["target"]
#
# x = x.astype('float32') / 255.0
# y = y.astype('int32')

x = np.array([[2.0, 3.0, -1.0], [3.0, -1.0, 0.5], [0.5, 1.0, 1.0], [1.0, 1.0, -1.0]])
y = np.array([1.0, -1.0, -1.0, 1.0])


n_layers = 2
hidden_layers_dim = [10]  # number of neurons in each hidden layer. len = n_layers - 1
#activation_funcs = ['relu', 'softmax']
activation_funcs = ['tanh', None]

np.random.seed(100)


# engine = NNEngine(x, y, n_layers, hidden_layers_dim, activation_funcs, classification=True)
# engine.init_params(classes=10)
engine = NNEngine(x, y, n_layers, hidden_layers_dim, activation_funcs, classification=False)
engine.init_params()


for k in range(100):
    # forward
    #cost, perf, y, y_pred = engine.forward_propagation(batch_size=64)
    cost, perf, y, y_pred = engine.forward_propagation()

    # backward
    engine.backward_propagation(y, y_pred)

    # update weights
    #learning_rate = 1.0 - 0.9 * k / 100
    learning_rate = 0.01
    engine.gradient_descent(learning_rate)

    #print(f"step {k}. cost={cost}; acc={perf}")
    print(f"step {k}. cost={cost}; y_pred={y_pred.ravel()}")

    engine.activations_values = None
