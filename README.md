# Neural Network - using only numpy

Builds a neural network engine from the ground up using only numpy. Implements backpropagation and gradient descent from matrices multiplications. You can input how many layers the NN is going to have, what is the activation function of each layer and how many units (neurons) each layer has. It works both for regression and classification problems.

I've tested this framework on the [MNIST dataset](https://en.wikipedia.org/wiki/MNIST_database). It worked pretty well, with ~90% of accuracy on out-of-sample data for only two hidden layers with 100 and 50 units, respectively.

## Project Structure
- `engine`:
  contains the engine itself that builds the neural network, does backpropagation and update the parameters (that's the *optimizer.py*). The training and validation parts are on *training.py*
- `use_cases`:
  few examples using the framework from *engine*

## Technologies
I only used Python and numpy. On `use_cases\mnist\demo_with_engine.py` I've used *sklearn*, but only to import the MNIST dataset - you could download the dataset for yourself and use it from your computer, so you wouldn't needed to import sklearn.

## How to run
To run the scripts that are in the subdirectory `use_cases`:
1. Clone the repository:
   ```bash
   git clone https://github.com/eriqsp/neural-network-scratch.git
   cd neural-network-scratch
2. Install dependencies
   ```bash
   pip install -r requirements.txt
3. Run and see the magic
   ```bash
   python use_cases\mnist\demo_with_engine.py


## Example
The simplest example for a regression problem:
```python
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

# get r2 for test set
t.validation(x, y)
```

## License

MIT
