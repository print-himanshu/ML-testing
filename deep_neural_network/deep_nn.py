import numpy as np

# --------------------Forward Propagation--------------------------


def initialize_parameters_two(n_x, n_h, n_y):
    """
    Argument:
    n_x -- size of the input layer
    n_h -- size of the hidden layer
    n_y -- size of the output layer

    Returns:
    parameters -- python dictionary containing your parameters:
                    W1 -- weight matrix of shape (n_h, n_x)
                    b1 -- bias vector of shape (n_h, 1)
                    W2 -- weight matrix of shape (n_y, n_h)
                    b2 -- bias vector of shape (n_y, 1)
    """
    np.random.seed(1)

    W1 = np.random.randn(n_h, n_x) * 0.01
    b1 = np.zeros((n_h, 1))
    W2 = np.random.randn(n_y, n_h) * 0.01
    b2 = np.zeros((n_y, 1))

    parameters = {
        "W1": W1,
        "b1": b1,
        "W2": W2,
        "b2": b2
    }

    assert(W1.shape == (n_h, n_x))
    assert(b1.shape == (n_h, 1))
    assert(W2.shape == (n_y, n_h))
    assert(b2.shape == (n_y, 1))

    return parameters


def initialize_paraeters_deep(layer_dims):
    """
    Arguments:
    layer_dims -- python array (list) containing the dimensions of each layer in our network

    Returns:
    parameters -- python dictionary containing your parameters "W1", "b1", ..., "WL", "bL":
                    Wl -- weight matrix of shape (layer_dims[l], layer_dims[l-1])
                    bl -- bias vector of shape (layer_dims[l], 1)
    """

    np.random.seed(3)
    parameters = {}
    L = len(layer_dims)

    for l in range(1, L):
        parameters[f"W{l}"] = np.random.randn(
            layer_dims[l], layer_dims[l-1]) / np.sqrt(layer_dims[l-1])
        parameters[f"b{l}"] = np.zeros((layer_dims[l], 1))

    assert(parameters[f"W{l}"].shape == (layer_dims[l], layer_dims[l-1]))
    assert(parameters[f"b{l}"].shape == (layer_dims[l], 1))

    return parameters


def linear_forward(A, W, b):
    """
    Implement the linear part of a layer's forward propagation.

    Arguments:
    A -- activations from previous layer (or input data): (size of previous layer, number of examples)
    W -- weights matrix: numpy array of shape (size of current layer, size of previous layer)
    b -- bias vector, numpy array of shape (size of the current layer, 1)

    Returns:
    Z -- the input of the activation function, also called pre-activation parameter 
    cache -- a python tuple containing "A", "W" and "b" ; stored for computing the backward pass efficiently
    """
    z = np.dot(W, A) + b

    assert(z.shape == (W.shape[0], A.shape[1]))
    cache = (A, W, b)

    return z, cache


def sigmoid(z):
    a = 1/(1 + np.exp(-z))
    cache = z

    return a, cache


def relu(z):
    a = np.maximum(0, z)
    cache = z

    assert(a.shape == z.shape)
    return a, cache


def linear_activation_forward(A_prev, W, b, activation="simgoid"):
    """
    Implement the forward propagation for the LINEAR->ACTIVATION layer

    Arguments:
    A_prev -- activations from previous layer (or input data): (size of previous layer, number of examples)
    W -- weights matrix: numpy array of shape (size of current layer, size of previous layer)
    b -- bias vector, numpy array of shape (size of the current layer, 1)
    activation -- the activation to be used in this layer, stored as a text string: "sigmoid" or "relu"

    Returns:
    A -- the output of the activation function, also called the post-activation value 
    cache -- a python tuple containing "linear_cache" and "activation_cache";
             stored for computing the backward pass efficiently
    """

    z, linear_cache = linear_forward(A_prev, W, b)
    if activation == "sigmoid":
        a, activation_cache = sigmoid(z)

    elif activation == 'relu':
        a, activation_cache = relu(z)

    cache = (linear_cache, activation_cache)

    return a, cache


def L_model_forward(X, parameters):
    """
    Implement forward propagation for the [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID computation

    Arguments:
    X -- data, numpy array of shape (input size, number of examples)
    parameters -- output of initialize_parameters_deep()

    Returns:
    AL -- last post-activation value
    caches -- list of caches containing:
                every cache of linear_activation_forward() (there are L-1 of them, indexed from 0 to L-1)
    """

    caches = []
    L = len(parameters) // 2
    A = X

    for l in range(1, L):
        A, cache = linear_activation_forward(
            A,
            parameters[f'W{l}'],
            parameters[f"b{l}"],
            activation='relu')

        caches.append(cache)

    A_L, cache = linear_activation_forward(
        A,
        parameters[f"W{L}"],
        parameters[f"b{L}"],
        activation='sigmoid'
    )

    caches.append(cache)

    assert(A_L.shape == (parameters[f"W{L}"].shape[0], X.shape[1]))
    return A_L, caches

# ----------------------Cost--------------------------------------


def cross_entropy_cost(A_L, y):
    y_if_1 = np.multiply(y, np.log(A_L))
    y_if_0 = np.multiply((1-y), np.log(1-A_L))

    loss = -(y_if_1 + y_if_0)
    m = y.shape[1]

    cost = (1./m) * np.sum(loss)
    cost = np.squeeze(cost)

    assert(cost.shape == ())
    return cost

# ---------------------Backward Propagation-----------------------


def linear_backward(dz, cache):
    # Here cache is "linear_cache" containing (A_prev, W, b) coming from the forward propagation in the current layer
    """
    Implement the linear portion of backward propagation for a single layer (layer l)

    Arguments:
    dZ -- Gradient of the cost with respect to the linear output (of current layer l)
    cache -- tuple of values (A_prev, W, b) coming from the forward propagation in the current layer

    Returns:
    dA_prev -- Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
    dW -- Gradient of the cost with respect to W (current layer l), same shape as W
    db -- Gradient of the cost with respect to b (current layer l), same shape as b
    """

    A_prev, W, b = cache
    m = A_prev.shape[1]

    dW = (1/m) * np.dot(dz, A_prev.T)
    db = (1/m) * np.sum(dz, axis=1, keepdims=True)
    da_prev = np.dot(W.T, dz)

    assert(dW.shape == W.shape)
    assert(b.shape == db.shape)
    assert(A_prev.shape == da_prev.shape)

    return da_prev, dW, db


def sigmoid_backward(da, activation_cache):
    z= activation_cache
    a, _ = sigmoid(z)
    dz = da * a * (1-a)

    assert (dz.shape == z.shape)
    return dz


def relu_backward(da, activation_cache):
    z = activation_cache
    dz = np.array(da, copy=True)
    dz[z <= 0] = 0

    assert (dz.shape == z.shape)
    return dz


def linear_activation_backward(da, cache, activation='sigmoid'):
    """
    Implement the backward propagation for the LINEAR->ACTIVATION layer.

    Arguments:
    dA -- post-activation gradient for current layer l 
    cache -- tuple of values (linear_cache, activation_cache) we store for computing backward propagation efficiently
    activation -- the activation to be used in this layer, stored as a text string: "sigmoid" or "relu"

    Returns:
    dA_prev -- Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
    dW -- Gradient of the cost with respect to W (current layer l), same shape as W
    db -- Gradient of the cost with respect to b (current layer l), same shape as b
    """
    linear_cache, activation_cache = cache

    if activation == "sigmoid":
        dz = sigmoid_backward(da, activation_cache)
    elif activation == 'relu':
        dz = relu_backward(da, activation_cache)

    da_prev, dW, db = linear_backward(dz, linear_cache)

    return da_prev, dW, db


def L_model_backward(A_L, y, caches):
    """
    Implement the backward propagation for the [LINEAR->RELU] * (L-1) -> LINEAR -> SIGMOID group

    Arguments:
    AL -- probability vector, output of the forward propagation (L_model_forward())
    Y -- true "label" vector (containing 0 if non-cat, 1 if cat)
    caches -- list of caches containing:
                every cache of linear_activation_forward() with "relu" (it's caches[l], for l in range(L-1) i.e l = 0...L-2)
                the cache of linear_activation_forward() with "sigmoid" (it's caches[L-1])

    Returns:
    grads -- A dictionary with the gradients
             grads["dA" + str(l)] = ... 
             grads["dW" + str(l)] = ...
             grads["db" + str(l)] = ... 
    """
    grads = {}
    L = len(caches)
    m = y.shape[1]

    da_L = - (np.divide(y, A_L) - np.divide(1-y, 1-A_L))

    current_cache = caches[L - 1]
    da, grads[f'dW{L}'], grads[f'db{L}'] = linear_activation_backward(
        da_L, current_cache)

    for l in reversed(range(1, L)):
        current_cache = caches[l-1]
        da, grads[f'dW{l}'], grads[f'db{l}'] = linear_activation_backward(
            da, current_cache, activation='relu')

    return grads

# ----------------------Update and Predict---------------------


def update_parameters(parameters, grads, learning_rate):
    """
    Update parameters using gradient descent

    Arguments:
    parameters -- python dictionary containing your parameters 
    grads -- python dictionary containing your gradients, output of L_model_backward

    Returns:
    parameters -- python dictionary containing your updated parameters 
                  parameters["W" + str(l)] = ... 
                  parameters["b" + str(l)] = ...
    """

    L = len(parameters) // 2

    for l in range(L):
        parameters[f"W{l+1}"] = parameters[f"W{l+1}"] - \
            learning_rate * grads[f"dW{l+1}"]
        parameters[f"b{l+1}"] = parameters[f"b{l+1}"] - \
            learning_rate * grads[f"db{l+1}"]

    return parameters


def predict(X, Y, parameters, text = "Training"):
    a2, caches = L_model_forward(X, parameters)

    m = Y.shape[1]
    output = np.zeros((1, m))

    output = (a2 > 0.5) * 1.0

    #print(f"Accuracy (1st method) = {100 - np.mean(np.abs(output - Y)) * 100}")
    print(f"{text} Accuracy ={(np.sum(Y == output) / m) * 100} ")

    return output


if __name__ == "__main__":
    pass
