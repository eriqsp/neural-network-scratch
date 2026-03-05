import numpy as np
from engine.training import Training

x = np.array([[2.0, 3.0, -1.0], [3.0, -1.0, 0.5], [0.5, 1.0, 1.0], [1.0, 1.0, -1.0]])
y = np.array([1.0, -1.0, -1.0, 1.0])


layers = [(10, 'tanh'), (1, None)]  # [(n_units, activation_function)]
classification = False
batch_size = None

t = Training(layers, classification, 10)

# update parameters with sgd (stochastic gradient descent) procedure
t.training_procedure(x, y, batch_size)

# get r2
t.validation(x, y)
