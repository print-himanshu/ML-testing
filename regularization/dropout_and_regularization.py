import numpy as np
import deep_nn as nn

# --------------------------------Regulariaztion------------------------------------------------------------


def cost_with_regulariztion(A3, y, parameters, lambda_var):
    """
    Implement the cost function with L2 regularization. See formula (2) above.

    Arguments:
    A3 -- post-activation, output of forward propagation, of shape (output size, number of examples)
    Y -- "true" labels vector, of shape (output size, number of examples)
    parameters -- python dictionary containing parameters of the model

    Returns:
    cost - value of the regularized loss function (formula (2))
    """

    m = y.shape[1]
    W1 = parameters['W1']
    W2 = parameters['W2']
    W3 = parameters['W3']

    cross_entropy = nn.cross_entropy_cost(A3, y)
    l2_regularization = lambda_var / \
        (2 * m) * (np.sum(np.square(W1)) +
                   np.sum(np.square(W2)) + np.sum(np.square(W3)))
    cost = cross_entropy + l2_regularization

    return cost


def linear_backward_with_regularization(dz, cache, lambda_var):
    a_prev, W, b = cache
    m = a_prev.shape[1]

    dW = (1/m) * np.dot(dz, a_prev.T) + (lambda_var / m) * W
    db = (1/m) * np.sum(dz, axis=1, keepdims=True)
    da_prev = np.dot(W.T, dz)

    assert(dW.shape == W.shape)
    assert(b.shape == db.shape)
    assert(a_prev.shape == da_prev.shape)

    return da_prev, dW, db


def linear_activation_backward_with_regularization(da, cache, lambda_var, activation='sigmoid'):
    linear_cache, activation_cache = cache

    if activation == 'sigmoid':
        dz = nn.sigmoid_backward(da, activation_cache)
    elif activation == 'relu':
        dz = nn.relu_backward(da, activation_cache)

    da_prev, dW, db = linear_backward_with_regularization(
        dz, linear_cache, lambda_var)

    return da_prev, dW, db


def L_model_backward_with_regulariztion(A_L, y, caches, lambda_var):

    grads = {}
    L = len(caches)
    m = y.shape[1]

    da_L = - (np.divide(y, A_L) - np.divide(1-y, 1-A_L))

    current_cache = caches[L - 1]
    da, grads[f'dW{L}'], grads[f'db{L}'] = linear_activation_backward_with_regularization(
        da_L, current_cache, lambda_var)

    for l in reversed(range(1, L)):
        current_cache = caches[l-1]
        da, grads[f'dW{l}'], grads[f'db{l}'] = linear_activation_backward_with_regularization(
            da, current_cache, lambda_var, activation='relu')

    return grads

# ---------------------------------Dropout-------------------------------------------------------------------

def linear_activation_forward_with_dropout(a_prev, w, b, keep_prob=0.5, activation='sigmoid'):
   
    z, linear_cache = nn.linear_forward(a_prev, w, b)

    if activation == 'sigmoid':
        a, a_cache = nn.sigmoid(z)

    elif activation == 'relu':
        a, a_cache = nn.relu(z)

    d = np.random.rand(z.shape[0], z.shape[1])
    d = (d < keep_prob).astype(int)

    a = np.multiply(a, d)
    a /= keep_prob

    activation_cache = [a_cache, d]
    cache = (linear_cache, activation_cache)

    return a, cache

def L_model_forward_with_dropout(X, parameters, keep_prob = 0.5, seed = 1):
    caches = []
    L = len(parameters) // 2
    A = X
    np.random.seed(seed)

    for l in range(1,L):
        A, cache = linear_activation_forward_with_dropout(
            A,
            parameters[f"W{l}"],
            parameters[f'b{l}'],
            keep_prob,
            activation = 'relu'
        )
        caches.append(cache)

    aL, cache = linear_activation_forward_with_dropout(
        A,
        parameters[f"W{L}"],
        parameters[f'b{L}'],
        keep_prob = 1
    )

    caches.append(cache)
    assert(aL.shape == (parameters[f"W{L}"].shape[0], X.shape[1]))
    return aL, caches


# ---------------------------------Backpropgation with dropout-----------------------------------------------
def linear_backward_with_dropout(dz,linear_cache, keep_prob):
    a_prev, W, b, d = linear_cache
    m = a_prev.shape[1]

    dW = (1/m) * np.dot(dz, a_prev.T)
    db = (1/m) * np.sum(dz, axis = 1, keepdims = True)
    da_prev = np.dot(W.T, dz)

    da_prev = np.multiply(da_prev, d)
    da_prev /= keep_prob

    assert(dW.shape == W.shape)
    assert(db.shape == b.shape)
    assert(da_prev.shape == a_prev.shape)

    return da_prev, dW, db





  

