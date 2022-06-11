import numpy as np
import deep_neural_network.deep_nn as nn

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


def dropout_linear_activation_forward(a_prev, W, b, keep_prob=0.5, activation='sigmoid'):
    """
    Implement one layer forward propgation 
    Input:
        a_prev: Input to the layer ; np_array of shape(n[l], m),        
        W     : Weight of the layer; np_array of shape(n[l], n[l-1]),
        b     : bias of the layer  ; np_array of shape(n[l], 1),
        keep_probs : Parameter controlling the % of the layer should be retained,
        activation : Determine the non linear activation of the layer ("sigmoid" or "relu")

    Output:
        a : Output the activation of the current layer
        cache: Dict containing every parammeter used for forward propagation later needed by backward
               propagation
        d: Dropout parameter containing of 1 and 0, to decide which layer to keep and which to drop
    """
    a, cache = nn.linear_activation_forward(a_prev, W, b, activation)

    d = np.random.rand(*a.shape)
    d = (d < keep_prob).astype(int)

    a = np.multiply(a, d)    # Applying dropout
    a /= keep_prob          # Inverted dropout

    return a, cache, d


def dropout_L_model_forward(X, parameters, keep_probs=0.5):
    """
    Implement forward propagation for the [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID computation

    Arguments:
    X -- data, numpy array of shape (input size, number of examples)
    parameters -- output of initialize_parameters_deep()
    keep_probs: % of the neurons to keep in the current layer

    Returns:
    AL -- last post-activation value
    caches -- list of caches containing:
                every cache of linear_activation_forward() (there are L-1 of them, indexed from 0 to L-1)
    """
    caches = []
    d_list = []
    L = len(parameters) // 2
    A = X

    for l in range(1, L):
        A, cache, d = dropout_linear_activation_forward(
            A,
            parameters[f"W{l}"],
            parameters[f'b{l}'],
            keep_probs,
            activation='relu'
        )
        caches.append(cache)
        d_list.append(d)

    aL, cache = nn.linear_activation_forward(
        A,
        parameters[f'W{L}'],
        parameters[f'b{L}'],
        activation = 'sigmoid'
    )
    caches.append(cache)

    assert(aL.shape == (parameters[f"W{L}"].shape[0], X.shape[1]))
    return aL, caches, d_list


# ---------------------------------Backpropgation with dropout-----------------------------------------------
def dropout_linear_backward(dz, linear_cache, d, keep_prob=0.5):
    a_prev, W, b = linear_cache
    da_prev, dW, db = nn.linear_backward(dz, linear_cache)

    da_prev = np.multiply(da_prev, d)
    da_prev /= keep_prob

    assert(da_prev.shape == a_prev.shape)

    return da_prev, dW, db


def dropout_linear_activation_backward(da, cache, d, keep_probs, activation='relu'):
    linear_cache, activation_cache = cache

    if activation == 'sigmoid':
        dz = nn.sigmoid_backward(da, activation_cache)
    elif activation == 'relu':
        dz = nn.relu_backward(da, activation_cache)

    da_prev, dW, db = dropout_linear_backward(
        dz, linear_cache, d, keep_probs)

    return da_prev, dW, db


def dropout_L_model_backward(AL, y, caches, d_list, keep_probs):
    L = len(caches)
    m = y.shape[1]
    grads = {}

    da_L = - (np.divide(y, AL) - np.divide(1-y, 1-AL))
    current_cache = caches[L-1]

    da, grads[f"dW{L}"], grads[f"db{L}"] = dropout_linear_activation_backward(
        da_L,
        current_cache,
        d_list.pop(),
        keep_probs,
        activation='sigmoid',
    )  # Last- Layer parammeter

    for i in reversed(range(2, L)):
        if len(d_list) == 0:
            print("d_list is empty\nCheck for L_model backward with dropout")

        current_cache = caches[i - 1]
        d = d_list.pop()

        da, grads[f"dW{i}"], grads[f"db{i}"] = dropout_linear_activation_backward(
            da,
            current_cache,
            d,
            keep_probs
        )  # all hidden layer except first hidden layer which has no d and keep_probs for calculation

    current_cache = caches[0]
    da, grads["dW1"], grads['db1'] = nn.linear_activation_backward(
        da,
        current_cache,
        activation='relu'
    ) # 1st hidden layer parammeter grads

    return grads

