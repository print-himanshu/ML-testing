from unittest import TestCase
import deep_nn as nn
import dropout_and_regularization as dr
import numpy as np
from numpy.testing import assert_almost_equal


class TestDropout(TestCase):
    def setUp(self):
        self.layer_dims = [3, 2, 3, 1]  # 3 layer nn
        self.X = np.array([
            [1.62434536, -0.61175641, -0.52817175, -1.07296862,  0.86540763],
            [-2.3015387,  1.74481176, -0.7612069,   0.3190391,  -0.24937038],
            [1.46210794, -2.06014071, -0.3224172,  -0.38405435,  1.13376944]])
        self.parameters = {
            'W1': np.array([
                [-1.09989127, -0.17242821, -0.87785842],
                [0.04221375,  0.58281521, -1.10061918]]),

            'b1': np.array([
                [1.14472371],
                [0.90159072]]),  # b1

            'W2': np.array([
                [0.50249434,  0.90085595],
                [-0.68372786, -0.12289023],
                [-0.93576943, -0.26788808]]),  # w2 (3,2)

            'b2':  np.array([
                [0.53035547],
                [-0.69166075],
                [-0.39675353]]),  # b2
            # w3
            'W3': np.array([[-0.6871727, -0.84520564, -0.67124613]]),
            'b3': np.array([[-0.0126646]])  # b3
        }
        self.cache = {
            'a0': self.X,
            'z1': np.array([
                [-1.52855314,  3.32524635,  2.13994541,  2.60700654, -0.75942115],
                [-1.98043538,  4.1600994,   0.79051021,  1.46493512, -0.45506242]]),  # z1
            'd1': np.array([
                [1, 0, 1, 1, 1],
                [1, 1, 1, 1, 1]]),  # d1

            'a1':  np.array([
                [0., 0., 3.05706487, 3.72429505, 0.],
                [0., 5.94299915, 1.1293003, 2.09276446, 0.]]),

            'z2':  np.array([
                [0.53035547,   5.88414161,   3.08385015, 4.28707196,  0.53035547],
                [-0.69166075, -1.42199726,  -2.92064114, -3.49524533, -0.69166075],
                [-0.39675353, -1.98881216,  -3.55998747, -4.44246165, -0.39675353]]),

            'd2': np.array([
                [1, 1, 1, 0, 1],
                [1, 1, 1, 1, 1],
                [0, 0, 1, 1, 0]]),   # d2

            'a2': np.array([
                [0.75765067, 8.40591658, 4.40550021, 0., 0.75765067],
                [0., 0., 0., 0., 0.],
                [0., 0., 0., 0., 0.]]),

            'z3': np.array(
                [[-0.53330145, -5.78898099, -3.04000407, -0.0126646, -0.53330145]]),  # z3
            'd3': np.array([[1, 1, 1, 1, 1]]),
            'aL': np.array(
                [[0.36974721, 0.00305176, 0.04565099, 0.49683389, 0.36974721]]),
            'keep_probs': 0.7,
            'seed': 1
        }
        self.dropout_cache = [
            [
                [
                    self.X,  # x
                    self.parameters['W1'],
                    self.parameters['b1'],
                ],
                    self.cache['z1'],
            ],  # cache - 1
            [
                [
                    self.cache['a1'],
                    self.parameters['W2'],
                    self.parameters['b2'],
                ],
                    self.cache['z2'],
            ],  # cache - 2
            [
                [
                    self.cache['a2'],
                    self.parameters['W3'],
                    self.parameters['b3'],
                ],
                    self.cache['z3'],
            ]  # cache - 3
        ]
        self.y = np.array([[1, 1, 0, 1, 0]])
        self.grads = {
            'dz3': np.array([
                [-0.67733606, -0.50316611,  0.00348883, -0.50316611,  0.32266394]]),
            'dW3': np.array([
                [-0.06951191,  0.,  0.]]),
            'db3': np.array([[-0.2715031]]),
            'da2': np.array([
                [0.58180856,  0., -0.00299679,  0., -0.27715731],
                [0.,  0.53159854, -0.,  0.53159854, -0.34089673],
                [0.,  0., -0.00292733,  0., -0.]]),
            'dz2': np.array([
                [0.58180856,  0., -0.00299679,  0., -0.27715731],
                [0.,  0., -0.,  0., -0.],
                [0.,  0., -0.,  0., -0.]]),
            'dW2': np.array([
                [-0.00256518, -0.0009476],
                [0.,  0.],
                [0.,  0.]]),
            'db2': np.array([[0.06033089],
                             [0.],
                             [0.]]),
            'da1': np.array([
                [0.36544439,  0., -0.00188233,  0., -0.17408748],
                [0.65515713,  0., -0.00337459,  0., -0.]]),
            'dz1': np.array([
                [0.,  0., -0.00188233,  0., -0.],
                [0.,  0., -0.00337459,  0., -0.]]),
            'dW1': np.array([
                [0.00019884, 0.00028657, 0.00012138],
                [0.00035647, 0.00051375, 0.00021761]]),
            'db1': np.array([
                [-0.00037647],
                [-0.00067492]])
        }
        self.cache_2 = {
            'a0': self.X,
            'a1': np.array([
                [0., 0., 4.27989081, 5.21401307, 0.],
                [0., 8.32019881, 1.58102041, 2.92987024, 0.]]),
            'a2': np.array([
                [1.06071093, 0., 8.21049603, 0., 1.06071093],
                [0., 0., 0., 0., 0.],
                [0., 0., 0., 0., 0.]]),
            'aL': np.array([[0.32266394, 0.49683389, 0.00348883, 0.49683389, 0.32266394]]),

            'z1': np.array([
                [-1.52855314,  3.32524635,  2.13994541,  2.60700654, -0.75942115],
                [-1.98043538,  4.1600994,  0.79051021,  1.46493512, -0.45506242]]),
            'z2': np.array([
                [0.53035547,  8.02565606,  4.10524802,  5.78975856,  0.53035547],
                [-0.69166075, -1.71413186, -
                 3.81223329, -4.61667916, -0.69166075],
                [-0.39675353, -2.62563561, -4.82528105, -6.0607449, -0.39675353]]),
            'z3': np.array([[-0.7415562, -0.0126646, -5.65469333, -0.0126646, -0.7415562]]),

            'd1': np.array([
                [1, 0,  1,  1,  1],
                [1,  1,  1,  1, 0]]),
            'd2': np.array([
                [1, 0,  1, 0,  1],
                [0,  1, 0,  1,  1],
                [0, 0,  1, 0, 0]]),
            'd3': np.array([[1, 1, 1, 1, 1]]),
            'keep_probs': 0.8,
            'seed': 'unknown'

        }
        self.dropout_cache_2 = [
            [
                [
                    self.X,  # x
                    self.parameters['W1'],
                    self.parameters['b1'],
                ],
                    self.cache_2['z1'],
            ],  # cache - 1
            [
                [
                    self.cache_2['a1'],
                    self.parameters['W2'],
                    self.parameters['b2'],

                ],
                    self.cache_2['z2'],
            ],  # cache - 2
            [
                [
                    self.cache_2['a2'],
                    self.parameters['W3'],
                    self.parameters['b3'],
                ],
                    self.cache_2['z3'],
            ]  # cache - 3
        ]
        self.d_list = [
            self.cache_2['d1'],
            self.cache_2['d2'],
            self.cache_2['d3'],
        ]
    # -------------------------------  Dropout Forward Propagation-------------------------

    def data_linear_activation_forward_with_dropout(self):
        input = [
            self.X,
            self.parameters['W1'],
            self.parameters['b1'],
            self.cache['keep_probs'],
            'relu']
        output = {
            'a1':   self.cache['a1'],
            'cache': self.dropout_cache[0]
        }

        return input, output

    def test_linear_activation_forward_with_dropout(self):
        input, output = self.data_linear_activation_forward_with_dropout()
        np.random.seed(1)
        a, cache, d = dr.linear_activation_forward_with_dropout(*input)

        assert_almost_equal(a, output['a1'])
        for output_cache, result_cache in zip(output['cache'], cache):
            for output_linear, result_linear in zip(output_cache[0], result_cache[0]):
                assert_almost_equal(output_linear, result_linear)

            for output_activation, result_activation in zip(output_cache[1], result_cache[1]):
                assert_almost_equal(output_activation, result_activation)

    def data_linear_activation_forward_with_dropout_2(self):
        input = [
            self.cache['a2'],
            self.parameters['W3'],
            self.parameters['b3'],
            1,
            'sigmoid']
        output = {
            'a3':   self.cache['aL'],
            'cache': self.dropout_cache[2]
        }

        return input, output

    def test_linear_activation_forward_with_dropout_2(self):
        input, output = self.data_linear_activation_forward_with_dropout_2()
        np.random.seed(1)
        a, cache, d = dr.linear_activation_forward_with_dropout(*input)

        assert_almost_equal(a, output['a3'])
        assert_almost_equal(cache[0][0], self.cache['a2'])
        assert_almost_equal(cache[0][1], self.parameters['W3'])
        assert_almost_equal(cache[0][2], self.parameters['b3'])
        assert_almost_equal(cache[1], self.cache['z3'])
        assert_almost_equal(d, self.cache['d3'])

    # -------------------------------  L model forward-------------------------------------
    def data_L_model_forward_with_dropout(self):
        input = [
            self.X,
            self.parameters,
            self.cache['keep_probs'],
        ]
        output = self.cache['aL']

        return input, output

    def test_L_model_forward_with_dropout(self):
        input, output = self.data_L_model_forward_with_dropout()
        aL, caches, d_list = dr.L_model_forward_with_dropout(*input)

        for l, cache in enumerate(caches):
            l = l + 1
            assert_almost_equal(cache[0][0], self.cache[f'a{l - 1}'])
            assert_almost_equal(cache[0][1], self.parameters[f'W{l}'])
            assert_almost_equal(cache[0][2], self.parameters[f'b{l}'])
            assert_almost_equal(cache[1], self.cache[f'z{l}'])

        for l, d in enumerate(d_list):
            l = l + 1
            assert_almost_equal(d, self.cache[f'd{l}'])

        assert_almost_equal(aL, output)
    # -------------------------------  linear backward-------------------------------------
    def data_linear_backward_with_dropout(self):
        input = [
            self.grads['dz3'],
            [
                self.cache_2['a2'],
                self.parameters['W3'],
                self.parameters['b3'],
            ],
            self.cache_2['d2'],
            self.cache_2['keep_probs']
        ]

        return input

    def test_linear_backward_with_dropout(self):
        input = self.data_linear_backward_with_dropout()
        da_prev, dW, db = dr.linear_backward_with_dropout(*input)

        assert_almost_equal(dW, self.grads['dW3'])
        assert_almost_equal(db, self.grads['db3'])
        assert_almost_equal(da_prev, self.grads['da2'])
    # -------------------------------  linear activation backward-------------------------------------
    def data_linear_activation_backward_with_dropout(self):
        self.grads['da3'] = - (np.divide(self.y, self.cache_2['aL']) - np.divide(1- self.y, 1- self.cache_2['aL']))
        input = [
            self.grads['da3'],
            self.dropout_cache_2[-1],
            self.cache_2['d2'],
            self.cache_2['keep_probs'],
            'sigmoid'
        ]

        return input

    def test_linear_activation_backward_with_dropout(self):
        input = self.data_linear_activation_backward_with_dropout()
        da_prev, dW, db = dr.linear_activation_backward_with_dropout(*input)

        assert_almost_equal(da_prev, self.grads['da2'])
        assert_almost_equal(dW, self.grads['dW3'])
        assert_almost_equal(db, self.grads['db3'])

    # -------------------------------  L model backward-------------------------------------
    def data_L_model_backward(self):
        input = [
            self.cache_2['aL'],
            self.y,
            self.dropout_cache_2,
            self.d_list,
            self.cache_2['keep_probs']
        ]

        return input

    def test_L_model_backward(self):
        input = self.data_L_model_backward()
        grads = dr.L_model_backward(*input)

        for key, value in grads.items():
            assert_almost_equal(value, self.grads[key])
