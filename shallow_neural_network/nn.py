from mimetypes import init
from turtle import back
import numpy as np


def layer_size(X,Y):
    n_x = X.shape[0]
    h_1 = 4
    n_y = Y.shape[0]

    return (n_x, h_1, n_y)

def initilaize_parameter(n_x, h_1, n_y):

    np.random.seed(2)

    W1 = np.random.randn(h_1, n_x) * 0.01
    b1 = np.zeros((h_1, 1))
    W2 = np.random.randn(n_y, h_1) * 0.01
    b2 = np.zeros((n_y,1))

    weight = {"W1": W1,
            "b1": b1,
            "W2": W2,
            "b2": b2}

    return weight

def sigmoid(value):
    return 1/(1 + np.exp(-value))


def forward_propagation(X, parameter):

    W1 = parameter['W1']
    b1 = parameter['b1']
    W2 = parameter['W2']
    b2 = parameter['b2']

    z1 = np.dot(W1, X) + b1
    a1 = np.tanh(z1)
    z2 = np.dot(W2, a1) + b2
    a2 = sigmoid(z2)


    assert (a2.shape == (1, X.shape[1]))

    cache = {'z1': z1,
             "a1": a1,
             "z2": z2,
             "a2": a2}

    return a2, cache

def compute_cost(a2, y):
    y_if_1 = np.multiply(y , np.log(a2))
    y_if_0 = np.multiply((1-y), np.log(1 - a2))

    loss = -(y_if_1 + y_if_0)
    m = y.shape[1]
    cost = (1/m) * np.sum(loss)
    cost = float(np.squeeze(cost))

    assert(isinstance(cost, float))
    
    return cost


def backward_propagation(X,Y, cache , parameters):
    
    # Extracting Weight
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']   

    # Extracting cache
    a1 = cache['a1']
    z1 = cache['z1']
    a2 = cache['a2']
    z2 = cache['z2']
    
    m = Y.shape[1]

    # Hidden layer to outupt layer weight derivative
    dz2 = a2 - Y
    dw2 = (1/m) * np.dot(dz2, a1.T)
    db2 = (1/m) * np.sum(dz2, axis = 1, keepdims = True)
    
    # Input Layer to Hidden Layer to weight derivative
    tanh_derivate = 1 - np.power(a1, 2)
    dz1 = np.multiply(np.dot(W2.T, dz2), tanh_derivate)
    dw1 = (1/m) * np.dot(dz1, X.T)
    db1 = (1/m) * np.sum(dz1, axis = 1, keepdims = True)

    grads = {'dw1': dw1,
             'db1': db1,
             'dw2': dw2,
             'db2': db2}

    return grads

def update_parameters(parameters, grads , learning_rate):
    
    #Extracting Weight
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']

    # Extracting grads
    dw1 = grads['dw1']
    db1 = grads['db1']
    dw2 = grads['dw2']
    db2 = grads['db2']

    W1 = W1 - learning_rate * dw1
    b1 = b1 - learning_rate * db1
    W2 = W2 - learning_rate * dw2
    b2 = b2 - learning_rate * db2
    
    
    parameters = {"W1": W1,
                  'b1': b1,
                  'W2': W2,
                  "b2": b2}


    return parameters


def nn_model(X,Y, learning_rate, num_iteration = 10000, print_cost = False):
    n_x, h_1, n_y = layer_size(X,Y)
    costs = []
    parameters = initilaize_parameter(n_x, h_1, n_y)

    for i in range(num_iteration):
        a2, cache = forward_propagation(X,parameters)
        cost = compute_cost(a2, Y)
        grads = backward_propagation(X,Y,cache, parameters)
        parameters = update_parameters(parameters, grads , learning_rate)


        costs.append(cost)
        if print_cost:
            if i  % 1000 == 0:
                print(f"Cost After iteration {i} : {cost}")

    
    return parameters, costs


def predict(parameters, X):
    a2, cache = forward_propagation(X, parameters)
    m = X.shape[1]
    predictions = np.zeros((1, m))
    predictions = (a2 >= 0.5) * 1.0

    assert (predictions.shape == (1,m))
    return predictions



if __name__ == '__main__':
    pass
    # np.random.seed(1)
    # X_assess = np.random.randn(2, 3)
    # Y_assess = (np.random.randn(1, 3) > 0)
    # parameters, costs = nn_model(X_assess, Y_assess, 1.2, num_iteration=10000, print_cost = True)