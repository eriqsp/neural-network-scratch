import numpy as np
from engine.training import Training
from sklearn.datasets import fetch_openml


# MNIST dataset; https://en.wikipedia.org/wiki/MNIST_database
mnist = fetch_openml('mnist_784', version=1, as_frame=False)

x, y = mnist["data"], mnist["target"]
x = x.astype('float32') / 255.0
y = y.astype('str')


layers = [(100, 'relu'), (50, 'relu'), (10, 'softmax')]  # [(n_units, activation_function)]
classification = True
batch_size = 500


t = Training(layers, classification, classes=np.unique(y), seed=10)

x_train, x_val, y_train, y_val = t.train_test_split(x, y)

# update parameters with sgd (stochastic gradient descent)
t.training_procedure(x_train, y_train, batch_size, n_iter=1000)

# get accuracy for test set
t.validation(x_val, y_val)
