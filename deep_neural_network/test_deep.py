import deep_nn as nn
import numpy as np
from unittest import TestCase
from numpy.testing import assert_almost_equal


class TestDeep(TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass


# ---------------------------------------Initialize Parameter two-----------

    def data_initialize_parameters_two(self):
        output = {
            "W1":   np.array([
                [0.01624345, -0.00611756, -0.00528172],
                [-0.01072969,  0.00865408, -0.02301539]]),
            "b1": np.array([
                [0.],
                [0.]]),
            "W2": np.array([
                [0.01744812, -0.00761207]]),
            "b2": np.array([[0.]]),
        }

        return output

    def test_initialize_parameters_two(self):
        output = self.data_initialize_parameters_two()
        result = nn.initialize_parameters_two(3, 2, 1)

        assert_almost_equal(result['W1'], output['W1'])
        assert_almost_equal(result['b1'], output['b1'])
        assert_almost_equal(result['W2'], output['W2'])
        assert_almost_equal(result['b2'], output['b2'])

# ---------------------------------------Initialize Parameter deep----------
    def data_initialize_parameters_deep(self):
        output = {
            'W1': np.array([[0.01788628,  0.0043651,   0.00096497, -0.01863493, -0.00277388],
                            [-0.00354759, -0.00082741, -
                           0.00627001, -0.00043818, -0.00477218],
                            [-0.01313865,  0.00884622, 0.00881318,
                           0.01709573,  0.00050034],
                            [-0.00404677, -0.0054536, -0.01546477, 0.00982367, -0.01101068]]),

            'b1': np.array([[0.],
                            [0.],
                            [0.],
                            [0.]]),

            'W2': np.array([[-0.01185047, -0.0020565,  0.01486148, 0.00236716],
                            [-0.01023785, -0.00712993, 0.00625245, -0.00160513],
                            [-0.00768836, -0.00230031, 0.00745056, 0.01976111]]),

            'b2': np.array([[0.],
                            [0.],
                            [0.]]),

        }

        layer_dims = [5, 4, 3]

        return layer_dims, output

    def test_initialize_paramters_deep(self):
        layer_dims, output = self.data_initialize_parameters_deep()
        result = nn.initialize_paraeters_deep(layer_dims)

        assert_almost_equal(output['W1'], result['W1'])
        assert_almost_equal(output['b1'], result['b1'])
        assert_almost_equal(output['W2'], result['W2'])
        assert_almost_equal(output['b2'], result['b2'])

# ---------------------------------------Linear Forward--------------------
    def data_linear_forward(self):
        np.random.seed(1)
        """
        X = np.array([[-1.02387576, 1.12397796],
                      [-1.62328545, 0.64667545],
                      [-1.74314104, -0.59664964]])
        W = np.array([[ 0.74505627, 1.97611078, -1.24412333]])
        b = np.array([[1]])
        """
        A = np.random.randn(3, 2)
        W = np.random.randn(1, 3)
        b = np.random.randn(1, 1)

        input = [A, W, b]
        output = np.array([[3.26295337, -1.23429987]])

        return input, output

    def test_linear_forward(self):
        input, output = self.data_linear_forward()
        result, _ = nn.linear_forward(*input)

        assert_almost_equal(output, result)

# ---------------------------------------Sigmoid---------------------------
    def test_sigmoid(self):
        result, _ = nn.sigmoid(np.array([0, 2]))
        output = np.array([0.5, 0.88079708])
        assert_almost_equal(result, output)


# ---------------------------------------Linear Activation Forward---------

    def data_linear_activation_forward(self):
        """
            X = np.array([[-1.02387576, 1.12397796],
                          [-1.62328545, 0.64667545],
                          [-1.74314104, -0.59664964]])
            W = np.array([[ 0.74505627, 1.97611078, -1.24412333]])
            b = 5
        """
        np.random.seed(2)
        A_prev = np.random.randn(3, 2)
        W = np.random.randn(1, 3)
        b = np.random.randn(1, 1)

        sigmoid = np.array([[0.96890023, 0.11013289]])
        relu = np.array([[3.43896131, 0.]])

        input = [A_prev, W, b]
        output = {'sigmoid': sigmoid, 'relu': relu}
        return input, output

    def test_linear_activation_forward(self):
        input, output = self.data_linear_activation_forward()
        sigmoid, _ = nn.linear_activation_forward(*input, activation="sigmoid")
        relu, _ = nn.linear_activation_forward(*input, activation='relu')

        assert_almost_equal(output["sigmoid"], sigmoid)
        assert_almost_equal(output['relu'], relu)

# ---------------------------------------L model Forward-------------------
    def data_L_model_forward(self):
        np.random.seed(6)
        X = np.random.randn(5, 4)
        W1 = np.random.randn(4, 5)
        b1 = np.random.randn(4, 1)
        W2 = np.random.randn(3, 4)
        b2 = np.random.randn(3, 1)
        W3 = np.random.randn(1, 3)
        b3 = np.random.randn(1, 1)

        parameters = {"W1": W1,
                      "b1": b1,
                      "W2": W2,
                      "b2": b2,
                      "W3": W3,
                      "b3": b3}

        input = [X, parameters]
        output = np.array([[0.03921668, 0.70498921, 0.19734387, 0.04728177]])

        return input, output

    def test_L_model_forward(self):
        input, output = self.data_L_model_forward()
        result, _ = nn.L_model_forward(*input)

        assert_almost_equal(result, output)

# --------------------------------------Cost Function----------------------
    def test_cost(self):
        input = [
            np.array([[.8, .9, 0.4]]), # A_L
            np.asarray([[1, 1, 0]])]   # y

        output =  0.2797765635793422

        result = nn.cost(*input)

        self.assertAlmostEqual(output, result)

# --------------------------------------Linear Backward--------------------
    def data_linear_backward(self):
        """
        z, linear_cache = (np.array([[-0.8019545 ,  3.85763489]]), (np.array([[-1.02387576,  1.12397796],
        [-1.62328545,  0.64667545],
        [-1.74314104, -0.59664964]]), np.array([[ 0.74505627,  1.97611078, -1.24412333]]), np.array([[1]]))
        """
        np.random.seed(1)
        dZ = np.random.randn(3, 4)
        A = np.random.randn(5, 4)
        W = np.random.randn(3, 5)
        b = np.random.randn(3, 1)
        linear_cache = (A, W, b)

        da_prev = np.array([[-1.15171336,  0.06718465, -0.3204696,   2.09812712],
                            [0.60345879, -3.72508701, 5.81700741, -3.84326836],
                            [-0.4319552, -1.30987417, 1.72354705, 0.05070578],
                            [-0.38981415, 0.60811244, -1.25938424, 1.47191593],
                            [-2.52214926,  2.67882552, -0.67947465,  1.48119548]])
        dW = np.array([[0.07313866, -0.0976715,  -0.87585828,  0.73763362,  0.00785716],
                       [0.85508818, 0.37530413, -0.59912655,
                           0.71278189, -0.58931808],
                       [0.97913304, -0.24376494, -0.08839671, 0.55151192, -0.10290907]])
        db = np.array([[-0.14713786],
                       [-0.11313155],
                       [-0.13209101]])

        input = [dZ, linear_cache]
        
        output = {'da_prev': da_prev,
                  'dW': dW,
                  'db': db}

        return input , output  

    def test_linear_backward(self):
        input, output = self.data_linear_backward()
        da_prev, dW, db = nn.linear_backward(*input)

        assert_almost_equal(da_prev, output['da_prev'])
        assert_almost_equal(dW, output['dW'])
        assert_almost_equal(db, output['db'])

# -------------------------------------Linear Activation Backward----------
    def data_linear_activation_backward(self):
        """
        aL, linear_activation_cache = (np.array([[ 3.1980455 ,  7.85763489]]), ((np.array([[-1.02387576,  1.12397796], [-1.62328545,  0.64667545],
                                       [-1.74314104, -0.59664964]]), np.array([[ 0.74505627,  1.97611078, -1.24412333]]), 5), np.array([[ 3.1980455 ,  7.85763489]])))
        """
        np.random.seed(2)
        dA = np.random.randn(1, 2)
        A = np.random.randn(3, 2)
        W = np.random.randn(1, 3)
        b = np.random.randn(1, 1)
        Z = np.random.randn(1, 2)
        linear_cache = (A, W, b)
        activation_cache = Z
        linear_activation_cache = (linear_cache, activation_cache)

        output = {"sigmoid": {"da_prev": np.array([[0.11017994,  0.01105339],
                                                   [0.09466817, 0.00949723],
                                                   [-0.05743092, -0.00576154]]),
                              "dW": np.array([[0.10266786,  0.09778551, -0.01968084]]),
                              "db": np.array([[-0.05729622]])
                              },

                  "relu": {"da_prev": np.array([[0.44090989, -0.],
                                                [0.37883606, -0.],
                                                [-0.2298228, 0.]]),
                           "dW":  np.array([[0.44513824, 0.37371418, -0.10478989]]),
                           "db": np.array([[-0.20837892]])
                           }
                  }
        input = [dA, linear_activation_cache]

        return input, output

    def test_linear_activation_backward(self):
        input, output = self.data_linear_activation_backward()
        sigmoid = nn.linear_activation_backward(*input)
        relu = nn.linear_activation_backward(*input, activation = 'relu')

        assert_almost_equal(output['sigmoid']['da_prev'], sigmoid[0])
        assert_almost_equal(output['sigmoid']['dW'], sigmoid[1])
        assert_almost_equal(output['sigmoid']['db'], sigmoid[2])

        assert_almost_equal(output['relu']['da_prev'], relu[0])
        assert_almost_equal(output['relu']['dW'], relu[1])
        assert_almost_equal(output['relu']['db'], relu[2])

# -------------------------------------L Model backward-------------------
    def data_L_model_backward(self):
        """
        X = np.random.rand(3,2)
        Y = np.array([[1, 1]])
        parameters = {'W1': np.array([[ 1.78862847,  0.43650985,  0.09649747]]), 'b1': np.array([[ 0.]])}

        aL, caches = (np.array([[ 0.60298372,  0.87182628]]), [((np.array([[ 0.20445225,  0.87811744],
            [ 0.02738759,  0.67046751],
            [ 0.4173048 ,  0.55868983]]),
        np.array([[ 1.78862847,  0.43650985,  0.09649747]]),
        np.array([[ 0.]])),
    np.array([[ 0.41791293,  1.91720367]]))])
    """
        np.random.seed(3)
        A2 = np.random.randn(1, 2)
        Y = np.array([[1, 0]])

        A0 = np.random.randn(4, 2)  # X
        W1 = np.random.randn(3, 4)
        b1 = np.random.randn(3, 1)
        Z1 = np.random.randn(3, 2)
        linear_cache_activation_1 = ((A0, W1, b1), Z1)

        A1 = np.random.randn(3, 2)
        W2 = np.random.randn(1, 3)
        b2 = np.random.randn(1, 1)
        Z2 = np.random.randn(1, 2)
        linear_cache_activation_2 = ((A1, W2, b2), Z2)

        caches = (linear_cache_activation_1, linear_cache_activation_2)

        input = [A2, Y , caches]

        output = {'da1': np.array([[0.12913162, -0.44014127],
                                  [-0.14175655,  0.48317296],
                                  [0.01663708, -0.05670698]]),
                  'dW2': np.array([[-0.39202432, -0.13325855, -0.04601089]]),
                  'db2': np.array([[0.15187861]]),
                  'da0': np.array([[0.,  0.52257901],
                                  [0., -0.3269206],
                                  [0., -0.32070404],
                                  [0., -0.74079187]]),
                  'dW1': np.array([[0.41010002, 0.07807203, 0.13798444, 0.10502167],
                                   [0., 0., 0., 0.],
                                   [0.05283652, 0.01005865, 0.01777766, 0.0135308]]),
                  'db1': np.array([[-0.22007063],
                                  [0.],
                                  [-0.02835349]])}

        return input , output

    def test_L_model_backward(self):
        input, output = self.data_L_model_backward()
        result = nn.L_model_backward(*input)

        assert_almost_equal(output['dW2'], result['dW2'])
        assert_almost_equal(output['db2'], result['db2'])
        assert_almost_equal(output['dW1'], result['dW1'])
        assert_almost_equal(output['db1'], result['db1'])    

# -------------------------------------Update Parameters-------------------
    def data_update_parameters(self):
        """
        parameters = {'W1': np.array([[ 1.78862847,  0.43650985,  0.09649747],
            [-1.8634927 , -0.2773882 , -0.35475898],
            [-0.08274148, -0.62700068, -0.04381817],
            [-0.47721803, -1.31386475,  0.88462238]]),
    'W2': np.array([[ 0.88131804,  1.70957306,  0.05003364, -0.40467741],
            [-0.54535995, -1.54647732,  0.98236743, -1.10106763],
            [-1.18504653, -0.2056499 ,  1.48614836,  0.23671627]]),
    'W3': np.array([[-1.02378514, -0.7129932 ,  0.62524497],
            [-0.16051336, -0.76883635, -0.23003072]]),
    'b1': np.array([[ 0.],
            [ 0.],
            [ 0.],
            [ 0.]]),
    'b2': np.array([[ 0.],
            [ 0.],
            [ 0.]]),
    'b3': np.array([[ 0.],
            [ 0.]])}
        grads = {'dW1': np.array([[ 0.63070583,  0.66482653,  0.18308507],
            [ 0.        ,  0.        ,  0.        ],
            [ 0.        ,  0.        ,  0.        ],
            [ 0.        ,  0.        ,  0.        ]]),
    'dW2': np.array([[ 1.62934255,  0.        ,  0.        ,  0.        ],
            [ 0.        ,  0.        ,  0.        ,  0.        ],
            [ 0.        ,  0.        ,  0.        ,  0.        ]]),
    'dW3': np.array([[-1.40260776,  0.        ,  0.        ]]),
    'da1': np.array([[ 0.70760786,  0.65063504],
            [ 0.17268975,  0.15878569],
            [ 0.03817582,  0.03510211]]),
    'da2': np.array([[ 0.39561478,  0.36376198],
            [ 0.7674101 ,  0.70562233],
            [ 0.0224596 ,  0.02065127],
            [-0.18165561, -0.16702967]]),
    'da3': np.array([[ 0.44888991,  0.41274769],
            [ 0.31261975,  0.28744927],
            [-0.27414557, -0.25207283]]),
    'db1': 0.75937676204411464,
    'db2': 0.86163759922811056,
    'db3': -0.84161956022334572}
        """
        np.random.seed(2)
        W1 = np.random.randn(3, 4)
        b1 = np.random.randn(3, 1)
        W2 = np.random.randn(1, 3)
        b2 = np.random.randn(1, 1)
        parameters = {"W1": W1,
                      "b1": b1,
                      "W2": W2,
                      "b2": b2}
        np.random.seed(3)
        dW1 = np.random.randn(3, 4)
        db1 = np.random.randn(3, 1)
        dW2 = np.random.randn(1, 3)
        db2 = np.random.randn(1, 1)
        grads = {"dW1": dW1,
                 "db1": db1,
                 "dW2": dW2,
                 "db2": db2}

        input = [parameters, grads, 0.1]

        output = {"W1": np.array([[-0.59562069, -0.09991781, -2.14584584,  1.82662008],
                                  [-1.76569676, -0.80627147,
                                      0.51115557, -1.18258802],
                                  [-1.0535704, -0.86128581, 0.68284052, 2.20374577]]),
                  "b1": np.array([[-0.04659241],
                                  [-1.28888275],
                                  [0.53405496]]),
                  "W2": np.array([[-0.55569196,  0.0354055,  1.32964895]]),
                  "b2": np.array([[-0.84610769]])}

        return input, output

    def test_update_parameters(self):
        input, output = self.data_update_parameters()
        result = nn.update_parameters(*input)


        assert_almost_equal(output['W1'], result['W1'])
        assert_almost_equal(output['b1'], result['b1'])
        assert_almost_equal(output['W2'], result['W2'])
        assert_almost_equal(output['b2'], result['b2'])