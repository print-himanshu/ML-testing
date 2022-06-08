import numpy as np
import h5py
import os

def z(theta, x):
    return np.dot(theta.T, x)

def sigmoid(value):
    return 1/(1 + np.exp(-value))

def J(y, a):
    m = y.shape[1]

    y_if_1 = np.multiply(y, np.log(a))
    y_if_0 = np.multiply((1 - y), np.log(1 - a))
    loss = -(y_if_0 + y_if_1)

    cost = (1/m) * np.sum(loss, axis=1)
    return cost

def gradient_differentiation(a, y, x):
    m = y.shape[1]
    return (1/m) * np.dot(x, (a - y).T)

def forward_propagation(theta, x, y,alpha , num_iterations, print_cost=False):
    costs = []
    for i in range(num_iterations):
        a = sigmoid(z(theta , x))
        cost = J(y,a)
        grads = gradient_differentiation(a,y,x)

        theta -= alpha * grads

        if i % 100 == 0:
            costs.append(cost)
            if print_cost:
                print ("Cost after iteration %i: %f" %(i, cost))
        

    return theta, costs

def predict(theta, x):
    m = x.shape[1]
    a = sigmoid(z(theta, x))
    
    y_pred = np.zeros((1,m))
    y_pred = (a >= 0.5) * 1.0

    assert (y_pred.shape == (1,m))
    return y_pred  




