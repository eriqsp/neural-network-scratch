import numpy as np
from typing import Union
from engine.optimizer import NNOptimizer


class Training(NNOptimizer):
    def __init__(self, layers, classification, classes=None, seed=10):
        super().__init__(layers, classification, classes)

        np.random.seed(seed)

    def training_procedure(self, x, y, batch_size: Union[int, None], n_iter: int = 100):
        self.init_params(x)

        for k in range(n_iter):
            # forward
            cost, perf, yb, y_pred = self.forward_propagation(x, y, batch_size=batch_size)

            # backward
            self.backward_propagation(yb, y_pred)

            # update weights
            learning_rate = 0.01
            self.gradient_descent(learning_rate)

            if self.classification:
                print(f"step {k}. cost={cost}; accuracy={perf}")
            else:
                print(f"step {k}. cost={cost}; r2={perf}")

            self.activations_values = None

    def validation(self, x: np.array, y: np.array):
        _, perf, _, y_pred = self.forward_propagation(x, y, batch_size=None)

        print('\n' + '-' * 100)
        print(f'Test performance')
        print('-' * 100)
        if self.classification:
            print(f"accuracy={perf}")
        else:
            print(f"r2={perf}")

    @staticmethod
    def train_test_split(x, y, ratio=0.75):
        train_ratio = ratio
        train_size = int(len(x) * train_ratio)

        indices = np.arange(len(x))
        np.random.shuffle(indices)

        train_idx, val_idx = indices[:train_size], indices[train_size:]

        x_train, x_val = x[train_idx], x[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        return x_train, x_val, y_train, y_val
