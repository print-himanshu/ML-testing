import numpy as np

def weight_initialization(layer_dims):
    np.random.seed(1)
    W = np.random.randn(1, layer_dims)
    b = np.zeros((1,1))

    return W, b  

def polynomial_feature(x, degree):
    x_list = None
    for l in range(degree):
        l += 1
        poly = np.power(x, l)
        if l == 1:
            x_list = poly
        else:
            x_list = np.vstack((x_list, poly))

    return x_list

def hypothesis(x_poly, W, b):
    z = np.dot(W, x_poly) + b
    return z

def mse_cost(z, y):
    m = y.shape[1]
    loss = np.power(z - y, 2)

    cost = 1/(2 * m) * np.sum(loss)
    cost = np.squeeze(cost)

    return cost

def gradient_descent(z, y, x_poly):
    m = y.shape[1]
    dz = z - y

    dW = (1/m) * np.dot(dz, x_poly.T)   # (1,10)
    db = (1/m) * np.sum(dz, axis = 1, keepdims = True)  #(1,1)
    
    return dW, db

def update_parameter(W, b, dW, db, alpha):
    W -= alpha * dW
    b -= alpha * db

    return W, b


if __name__ == '__main__':
    pass
