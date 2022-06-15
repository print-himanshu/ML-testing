import numpy as np
from math import floor
import sklearn.datasets as sd
import matplotlib.pyplot as plt


def mini_batch_random(X, y, mini_batch_size=64, seed=0):
    np.random.seed(seed)

    m = X.shape[1]
    mini_batches = []

    permutation = list(np.random.permutation(m))
    shuffled_x = X[:, permutation]
    shuffled_y = y[:, permutation]

    complete_batch_count = floor(m / mini_batch_size)
    for epoch in range(complete_batch_count):

        mini_batch_x = shuffled_x[:, epoch *
                                  mini_batch_size:  (epoch + 1) * mini_batch_size]
        mini_batch_y = shuffled_y[:, epoch *
                                  mini_batch_size:  (epoch + 1) * mini_batch_size]

        mini_batch = (mini_batch_x, mini_batch_y)
        mini_batches.append(mini_batch)

    if m % mini_batch_size != 0:
        mini_batch_x = shuffled_x[:, complete_batch_count * mini_batch_size:]
        mini_batch_y = shuffled_y[:, complete_batch_count * mini_batch_size:]

        mini_batch = (mini_batch_x, mini_batch_y)
        mini_batches.append(mini_batch)

    return mini_batches


def initialize_velocity(parameters):
    L = len(parameters) // 2
    v = {}

    for l in range(L):
        l += 1
        v[f"dW{l}"] = np.zeros(parameters[f"W{l}"].shape)
        v[f"db{l}"] = np.zeros(parameters[f"b{l}"].shape)

    return v


def update_parameters_with_velocity(parameters, grads, v, beta1, alpha):
    """
    Update parameters using Momentum

    Arguments:
    parameters -- python dictionary containing your parameters:
                    parameters['W' + str(l)] = Wl
                    parameters['b' + str(l)] = bl
    grads -- python dictionary containing your gradients for each parameters:
                    grads['dW' + str(l)] = dWl
                    grads['db' + str(l)] = dbl
    v -- python dictionary containing the current velocity:
                    v['dW' + str(l)] = ...
                    v['db' + str(l)] = ...
    beta -- the momentum hyperparameter, scalar
    learning_rate -- the learning rate, scalar

    Returns:
    parameters -- python dictionary containing your updated parameters 
    v -- python dictionary containing your updated velocities
    """

    L = len(parameters) // 2

    for l in range(L):
        l += 1

        v[f'dW{l}'] = beta1 * v[f'dW{l}'] + (1 - beta1) * grads[f"dW{l}"]
        v[f'db{l}'] = beta1 * v[f'db{l}'] + (1 - beta1) * grads[f"db{l}"]

        parameters[f"W{l}"] -= alpha * v[f"dW{l}"]
        parameters[f"b{l}"] -= alpha * v[f"db{l}"]

    return parameters, v


def initialize_adam(parameters):
    """
    Initializes v and s as two python dictionaries with:
                - keys: "dW1", "db1", ..., "dWL", "dbL" 
                - values: numpy arrays of zeros of the same shape as the corresponding gradients/parameters.

    Arguments:
    parameters -- python dictionary containing your parameters.
                    parameters["W" + str(l)] = Wl
                    parameters["b" + str(l)] = bl

    Returns: 
    v -- python dictionary that will contain the exponentially weighted average of the gradient.
                    v["dW" + str(l)] = ...
                    v["db" + str(l)] = ...
    s -- python dictionary that will contain the exponentially weighted average of the squared gradient.
                    s["dW" + str(l)] = ...
                    s["db" + str(l)] = ...

    """

    v = {}
    s = {}
    L = len(parameters) // 2

    for l in range(L):
        l += 1
        v[f"dW{l}"] = np.zeros(parameters[f'W{l}'].shape)
        v[f"db{l}"] = np.zeros(parameters[f'b{l}'].shape)

        s[f"dW{l}"] = np.zeros(parameters[f'W{l}'].shape)
        s[f"db{l}"] = np.zeros(parameters[f'b{l}'].shape)

    return v, s


def update_parameters_with_adam(parameters, grads, v, s, epoch, alpha = 0.01, beta1=0.9, beta2=0.999, epsilon1=1e-8):
    """
    Update parameters using Adam
    
    Arguments:
    parameters -- python dictionary containing your parameters:
                    parameters['W' + str(l)] = Wl
                    parameters['b' + str(l)] = bl
    grads -- python dictionary containing your gradients for each parameters:
                    grads['dW' + str(l)] = dWl
                    grads['db' + str(l)] = dbl
    v -- Adam variable, moving average of the first gradient, python dictionary
    s -- Adam variable, moving average of the squared gradient, python dictionary
    epoch -- Current iteration number
    alpha -- the learning rate, scalar.
    beta1 -- Exponential decay hyperparameter for the first moment estimates 
    beta2 -- Exponential decay hyperparameter for the second moment estimates 
    epsilon1 -- hyperparameter preventing division by zero in Adam updates

    Returns:
    parameters -- python dictionary containing your updated parameters 
    v -- Adam variable, moving average of the first gradient, python dictionary
    s -- Adam variable, moving average of the squared gradient, python dictionary
    """

    L = len(parameters) // 2
    v_corrected = {}
    s_corrected = {}

    for l in range(L):
        l += 1
        # Momentum
        v[f'dW{l}'] = beta1 * v[f'dW{l}'] + (1 - beta1) * grads[f"dW{l}"]
        v[f'db{l}'] = beta1 * v[f'db{l}'] + (1 - beta1) * grads[f"db{l}"]

        # RMS prop
        s[f"dW{l}"] = beta2 * s[f"dW{l}"] + (1 - beta2) * np.square(grads[f"dW{l}"])
        s[f"db{l}"] = beta2 * s[f"db{l}"] + (1 - beta2) * np.square(grads[f"db{l}"])

        # Bias Correctness
        v_corrected[f"dW{l}"] = v[f'dW{l}']/ (1 - beta1 ** epoch)
        v_corrected[f"db{l}"] = v[f'db{l}']/ (1 - beta1 ** epoch)

        s_corrected[f'dW{l}'] = s[f"dW{l}"]/ (1 - beta2 ** epoch)
        s_corrected[f'db{l}'] = s[f"db{l}"]/ (1 - beta2 ** epoch)

        #Update
        parameters[f"W{l}"] -= alpha * (v_corrected[f"dW{l}"]/ (np.sqrt(s_corrected[f"dW{l}"] ) + epsilon1))
        parameters[f"b{l}"] -= alpha * (v_corrected[f"db{l}"]/ (np.sqrt(s_corrected[f"db{l}"] ) + epsilon1))


    return parameters, v, s

def load_dataset():
    np.random.seed(3)
    train_X, train_Y = sd.make_moons(n_samples=300, noise=.2) #300 #0.2 
    # Visualize the data
    plt.scatter(train_X[:, 0], train_X[:, 1], c=train_Y, s=40, cmap=plt.cm.Spectral)
    train_X = train_X.T
    train_Y = train_Y.reshape((1, train_Y.shape[0]))
    
    return train_X, train_Y

if __name__ == '__main__':
    x, y = load_dataset()
