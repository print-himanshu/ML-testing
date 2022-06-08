import numpy as np
from numpy.testing import assert_almost_equal
import dropout_and_regularization as dr
import deep_nn as nn
from unittest import TestCase


class TestRegularization(TestCase):
    def setUp(self):
        self.m = 5
        self.layer_dims = [3, 2, 3, 1]  # 3 layer nn
        self.lambda_var = 0.7

        self.X = np.array([
            [1.62434536, -0.61175641, -0.52817175, -1.07296862,  0.86540763],
            [-2.3015387,  1.74481176, -0.7612069,  0.3190391, -0.24937038],
            [1.46210794, -2.06014071, -0.3224172, -0.38405435,  1.13376944]])  # np.random.randn(3,5)

        self.parameters = {
            'W1': np.array([[-1.09989127, -0.17242821, -0.87785842],
                            [0.04221375,  0.58281521, -1.10061918]]),
            'b1': np.array([[1.14472371],
                            [0.90159072]]),
            'W2': np.array([[0.50249434,  0.90085595],
                            [-0.68372786, -0.12289023],
                            [-0.93576943, -0.26788808]]),
            'b2': np.array([[0.53035547],
                            [-0.69166075],
                            [-0.39675353]]),
            'W3': np.array([[-0.6871727, -0.84520564, -0.67124613]]),
            'b3': np.array([[-0.0126646]])}  # np.random.randn()

        self.y = np.array([[1, 1, 0, 1, 0]])

        self.a_L = np.array(
            [[0.40682402,  0.01629284,  0.16722898, 0.10118111,  0.40682402]])  # a3 (1,5)

        self.y = np.array([[1, 1, 0, 1, 0]])

        self.caches = [
            [
                [self.X,
                 np.array([
                     [-1.09989127, -0.17242821, -0.87785842],
                     [0.04221375,  0.58281521, -1.10061918]]),  # w1  (2,3)
                 np.array([
                     [1.14472371],
                     [0.90159072]]),  # b1       (2,1)
                 ],

                np.array([
                    [-1.52855314,  3.32524635,  2.13994541,
                     2.60700654, -0.75942115],
                    [-1.98043538,  4.1600994,   0.79051021,  1.46493512, -0.45506242]]),  # z1 (2,5)]]

            ],  # cache - 1
            [
                [
                    np.array([
                        [0.,  3.32524635,  2.13994541,  2.60700654,  0.],
                        [0.,  4.1600994,  0.79051021,  1.46493512,  0.]]),  # a1 (2,5)
                    np.array([
                        [0.50249434,  0.90085595],
                        [-0.68372786, -0.12289023],
                        [-0.93576943, -0.26788808]]),  # w2 (3,2)
                    np.array([
                        [0.53035547],
                        [-0.69166075],
                        [-0.39675353]]),  # b2       (3,1)
                ],

                np.array([[0.53035547,  5.94892323,  2.31780174,  3.16005701,  0.53035547],
                          [-0.69166075, -3.47645987, - \
                              2.25194702, -2.65416996, -0.69166075],
                          [-0.39675353, -4.62285846, -2.61101729, -3.22874921, -0.39675353]]),  # z2    (3,5)

            ],  # cache -2
            [
                [
                    np.array([
                        [0.53035547,  5.94892323,  2.31780174,
                         3.16005701,  0.53035547],
                        [0.,  0.,  0.,  0.,  0.],
                        [0.,  0.,  0.,  0.,  0.]]),  # a2    (3,5)
                    np.array([
                        [-0.6871727, -0.84520564, -0.67124613]]),  # w3      (1,3)
                    np.array([
                        [-0.0126646]])  # b3     (1,1)
                ],

                # z3 (1,5)
                np.array(
                    [[-0.3771104, -4.10060224, -1.60539468, -2.18416951, -0.3771104]]),

            ]
        ]

        self.grads = {
            'dZ3': np.array([[-0.59317598, -0.98370716,  0.16722898, -0.89881889,  0.40682402]]),
            'dW3': np.array([[-1.77691347, -0.11832879, -0.09397446]]),
            'db3': np.array([[-0.38032981]]),
            'dA2': np.array([[0.40761434,  0.67597671, -0.11491519,  0.6176438, -0.27955836],
                             [0.50135568,  0.83143484, -0.14134288,
                                 0.7596868, -0.34384996],
                             [0.39816708,  0.66030962, -0.11225181,  0.6033287, -0.27307905]]),
            'dZ2': np.array([[0.40761434,  0.67597671, -0.11491519,  0.6176438, -0.27955836],
                             [0.,  0., -0.,  0., -0.],
                             [0.,  0., -0.,  0., -0.]]),
            'dW2': np.array([[0.79276486,  0.85133918],
                             [-0.0957219, -0.01720463],
                             [-0.13100772, -0.03750433]]),
            'db2': np.array([[0.26135226],
                             [0.],
                             [0.]]),
            'dA1': np.array([[0.2048239,  0.33967447, -0.05774423,  0.31036252, -0.14047649],
                             [0.3672018,  0.60895764, -0.10352203,  0.5564081, -0.25184181]]),
            'dZ1': np.array([[0.,  0.33967447, -0.05774423,  0.31036252, -0.],
                             [0.,  0.60895764, -0.10352203,  0.5564081, -0.]]),
            'dW1': np.array([[-0.25604646,  0.12298827, -0.28297129],
                             [-0.17706303,  0.34536094, -0.4410571]]),
            'db1': np.array([[0.11845855],
                             [0.21236874]])}



    def tearDown(self):
        pass

# --------------------------------- l2 regulariztion ------------------------------------
    def data_cost_with_regularization(self):
        np.random.seed(1)
        parameters = {}
        L = len(self.layer_dims)
        for l in range(1, L):
            parameters[f'W{l}'] = np.random.randn(
                self.layer_dims[l], self.layer_dims[l-1])
            parameters[f'b{l}'] = np.random.randn(self.layer_dims[l], 1)

        input = [self.a_L, self.y, parameters, 0.1]
        output = 1.7864859451590758

        return input, output

    def test_cost_with_regularization(self):
        input, output = self.data_cost_with_regularization()
        result = dr.cost_with_regulariztion(*input)

        self.assertAlmostEqual(output, result)

# -------------------------------- L model forward--------------------------
    def data_L_model_forward(self):
        input = [self.X, self.parameters]

        output = {
            'a_L': self.a_L,
            'caches': self.caches
        }

        return input, output

    def test_L_model_forward(self):
        input, output = self.data_L_model_forward()
        a_L, caches = nn.L_model_forward(*input)

        assert_almost_equal(a_L, output['a_L'])

        for output_cache, result_cache in zip(output['caches'], caches):
            for output_linear, result_linear in zip(output_cache[0], result_cache[0]):
                assert_almost_equal(output_linear, result_linear)

            assert_almost_equal(output_cache[1], result_cache[1])

# --------------------------------  backward propagation with l2 regularization --------

    def data_linear_backward_with_regularization(self):

        dz3 = np.array(
            [[-0.59317598, -0.98370716,  0.16722898, -0.89881889,  0.40682402]])

        cache = [
            np.array([
                [0.53035547,  5.94892323,  2.31780174,
                 3.16005701,  0.53035547],
                [0.,  0.,  0.,  0.,  0.],
                [0.,  0.,  0.,  0.,  0.]]),  # a2    (3,5)
            np.array([
                [-0.6871727, -0.84520564, -0.67124613]]),  # w3      (1,3)
            np.array([
                [-0.0126646]])  # b3     (1,1)
        ]

        output = {
            'dW3': np.array([[-1.77691347, -0.11832879, -0.09397446]]),
            'db3': np.array([[-0.38032981]]),
            'dA2': np.array([[0.40761434,  0.67597671, -0.11491519,  0.6176438, -0.27955836],
                             [0.50135568,  0.83143484, -0.14134288,
                                 0.7596868, -0.34384996],
                             [0.39816708,  0.66030962, -0.11225181,  0.6033287, -0.27307905]])
        }

        input = [dz3, cache, self.lambda_var]

        return input, output

    def test_linear_backward_with_regularization(self):
        input, output = self.data_linear_backward_with_regularization()
        da_prev, dW, db = dr.linear_backward_with_regularization(*input)

        assert_almost_equal(output['dA2'], da_prev)
        assert_almost_equal(output['dW3'], dW)
        assert_almost_equal(output['db3'], db)

# --------------------------------  backward activation propgation with l2--------------
    def data_linear_activation_backward_with_regularization(self):

        da3 = - (np.divide(self.y, self.a_L) - np.divide(1-self.y, 1-self.a_L))

        cache = [
            [
                np.array([
                    [0.53035547,  5.94892323,  2.31780174,
                         3.16005701,  0.53035547],
                    [0.,  0.,  0.,  0.,  0.],
                    [0.,  0.,  0.,  0.,  0.]]),  # a2    (3,5)
                np.array([
                    [-0.6871727, -0.84520564, -0.67124613]]),  # w3      (1,3)
                np.array([
                    [-0.0126646]])  # b3     (1,1)
            ],

            # z3 (1,5)
            np.array(
                [[-0.3771104, -4.10060224, -1.60539468, -2.18416951, -0.3771104]]),

        ]

        input = [da3, cache, self.lambda_var, 'sigmoid']

        output = {
            'dW3': np.array([[-1.77691347, -0.11832879, -0.09397446]]),
            'db3': np.array([[-0.38032981]]),
            'dA2': np.array([[0.40761434,  0.67597671, -0.11491519,  0.6176438, -0.27955836],
                             [0.50135568,  0.83143484, -0.14134288,
                                 0.7596868, -0.34384996],
                             [0.39816708,  0.66030962, -0.11225181,  0.6033287, -0.27307905]])
        }

        return input, output

    def test_linear_activation_backward_with_regularization(self):
        input, output = self.data_linear_activation_backward_with_regularization()
        da_prev, dW, db = dr.linear_activation_backward_with_regularization(
            *input)

        assert_almost_equal(output['dA2'], da_prev, decimal=5)
        assert_almost_equal(output['dW3'], dW, decimal=5)
        assert_almost_equal(output['db3'], db, decimal=5)


# --------------------------------  L model backward------------------------------------

    def data_L_model_backward(self):
        input = [self.a_L, self.y, self.caches, self.lambda_var]
        output = self.grads

        return input, output

    def test_L_model_backward(self):
        input, output = self.data_L_model_backward()
        result = dr.L_model_backward(*input)

        for key in result.keys():
            assert_almost_equal(output[key], result[key], decimal=6)


