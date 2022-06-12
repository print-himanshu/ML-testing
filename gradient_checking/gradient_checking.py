from deep_neural_network import deep_nn as nn
import numpy as np


def dict_to_vector_main(key, value, keys, vector_output, count=0):
    new_vector = value.ravel()
    new_vector = np.expand_dims(new_vector, axis=1)
    keys += [key] * new_vector.shape[0]

    if count == 0:
        vector_output = new_vector
    else:
        vector_output = np.concatenate((vector_output, new_vector), axis=0)

    count += 1

    return keys, vector_output, count


def dictionary_to_vector(parameters):
    keys = []
    count = 0
    theta = np.array([])

    for key, value in parameters.items():
        keys, theta, count = dict_to_vector_main(key, value, keys, theta, count)

    return theta, keys


def vector_to_dictionary(theta, layer_dims):
    L = len(layer_dims)
    parameters = {}
    idx = 0  # theta_idx
    for l in range(1, L):
        w_dims = layer_dims[l] * layer_dims[l-1]
        b_dims = layer_dims[l]

        w = theta[idx: idx + w_dims]
        w_shape = (layer_dims[l], layer_dims[l-1])
        parameters[f'W{l}'] = np.reshape(w, w_shape)
        idx += w_dims

        b = theta[idx: idx + b_dims]
        b_shape = (layer_dims[l], 1)
        parameters[f"b{l}"] = np.reshape(b, b_shape)
        idx += b_dims

    return parameters


def gradient_to_vector(gradients, layer_dims):
    count = 0
    keys = []
    grads = np.array([])

    L = len(layer_dims)
    for l in range(1, L):
        key_w = f"dW{l}"
        value_w = gradients[key_w]
        keys, grads, count = dict_to_vector_main(key_w, value_w, keys, grads, count)

        key_b = f"db{l}"
        value_b = gradients[key_b]
        keys, grads, count = dict_to_vector_main(key_b, value_b, keys, grads, count)

    return grads, keys
