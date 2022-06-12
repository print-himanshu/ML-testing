import numpy as np
from numpy.testing import assert_almost_equal
import gradient_checking as gc
from unittest import TestCase


class Test_Gradient(TestCase):
    def setUp(self):
        self.X = np.array([
            [1.62434536, -0.61175641, -0.52817175],
            [-1.07296862,  0.86540763, -2.3015387],
            [1.74481176, -0.7612069,  0.3190391],
            [-0.24937038,  1.46210794, -2.06014071]])
        self.y = np.array([1, 1, 0])
        self.parameters = {
            'W1': np.array([
                [-0.3224172, -0.38405435,  1.13376944, -1.09989127],
                [-0.17242821, -0.87785842,  0.04221375,  0.58281521],
                [-1.10061918,  1.14472371,  0.90159072,  0.50249434],
                [0.90085595, -0.68372786, -0.12289023, -0.93576943],
                [-0.26788808,  0.53035547, -0.69166075, -0.39675353]]),
            'b1': np.array([
                [-0.6871727],
                [-0.84520564],
                [-0.67124613],
                [-0.0126646],
                [-1.11731035]]),
            'W2': np.array([
                [0.2344157,  1.65980218,  0.74204416, -0.19183555, -0.88762896],
                [-0.74715829,  1.6924546,  0.05080775, -0.63699565,  0.19091548],
                [2.10025514,  0.12015895,  0.61720311,  0.30017032, -0.35224985]]),
            'b2': np.array([
                [-1.1425182],
                [-0.34934272],
                [-0.20889423]]),
            'W3': np.array([[0.58662319, 0.83898341, 0.93110208]]),
            'b3': np.array([[0.28558733]])
        }
        self.layer_dims = [4, 5, 3, 1]
        self.theta = np.array([
            [-0.3224172],
            [-0.38405435],
            [1.13376944],
            [-1.09989127],
            [-0.17242821],
            [-0.87785842],
            [0.04221375],
            [0.58281521],
            [-1.10061918],
            [1.14472371],
            [0.90159072],
            [0.50249434],
            [0.90085595],
            [-0.68372786],
            [-0.12289023],
            [-0.93576943],
            [-0.26788808],
            [0.53035547],
            [-0.69166075],
            [-0.39675353],
            [-0.6871727],
            [-0.84520564],
            [-0.67124613],
            [-0.0126646],
            [-1.11731035],
            [0.2344157],
            [1.65980218],
            [0.74204416],
            [-0.19183555],
            [-0.88762896],
            [-0.74715829],
            [1.6924546],
            [0.05080775],
            [-0.63699565],
            [0.19091548],
            [2.10025514],
            [0.12015895],
            [0.61720311],
            [0.30017032],
            [-0.35224985],
            [-1.1425182],
            [-0.34934272],
            [-0.20889423],
            [0.58662319],
            [0.83898341],
            [0.93110208],
            [0.28558733]])
        self.grads = {
            'dZ3': np.array([
                [-0.02793328, -0.33416743,  0.99887329]]),
            'dW3': np.array([
                [0., 0., 2.24404238]]),
            'db3': np.array([[0.21225753]]),
            'dA2': np.array([
                [-0.01638631, -0.19603037,  0.58596224],
                [-0.02343555, -0.28036093,  0.83803812],
                [-0.02600873, -0.31114399,  0.930053]]),
            'dZ2': np.array([
                [-0., -0.,  0.],
                [-0., -0.,  0.],
                [-0.02600873, -0.31114399,  0.930053]]),
            'dW2': np.array([
                [0.,  0.,  0.,  0.,  0.],
                [0.,  0.,  0.,
                 0.,  0.],
                [0.91580165,  0.02451548, -0.10797954,  0.90281891,  0.]]),
            'db2': np.array([
                [0.],
                [0.],
                [0.19763343]]),
            'dA1': np.array([
                [-0.05462497, -0.65348177,  1.95334859],
                [-0.00312518, -0.03738674,  0.11175419],
                [-0.01605267, -0.19203904,  0.5740316],
                [-0.00780705, -0.09339619,  0.27917431],
                [0.00916157,  0.10960042, -0.32761103]]),
            'dZ1': np.array([
                [-0.05462497, -0.,  1.95334859],
                [-0., -0., 0.11175419],
                [-0., -0.19203904,  0.],
                [-0.00780705, -0.,  0.27917431],
                [0.,  0., -0.]]),
            'dW1': np.array([
                [-0.37347779, -1.47903216,  0.17596143, -1.33685036],
                [-0.01967514, -0.08573553,  0.01188465, -0.07674312],
                [0.03916037, -0.05539735,  0.04872715, -0.09359393],
                [-0.05337778, -0.21138458,  0.02514856, -0.19106384],
                [0.,  0.,  0.,  0.]]),
            'db1': np.array([
                [0.63290787],
                [0.0372514],
                [-0.06401301],
                [0.09045575],
                [0.]])
        }

    def tearDown(self):
        pass

    # ---------------------- Dictionary to vector ------------------------
    def data_dictionary_to_vector(self):
        keys = ['W1'] * 20 + ['b1'] * 5 + ['W2'] * \
            15 + ['b2'] * 3 + ['W3'] * 3 + ['b3'] * 1
        output = {
            'keys': keys,
            'theta': self.theta
        }
        return output

    def test_dictionary_to_vector(self):
        output = self.data_dictionary_to_vector()
        theta, keys = gc.dictionary_to_vector(self.parameters)

        assert_almost_equal(theta, output['theta'])
        self.assertListEqual(keys, output['keys'])

    # ---------------------- Vector to Dictionary ------------------------
    def test_vector_to_dictionary(self):
        parameters = gc.vector_to_dictionary(self.theta, self.layer_dims)

        for key, value in parameters.items():
            assert_almost_equal(parameters[key], self.parameters[key])

    # ---------------------- Gradients  to vector ------------------------
    def data_gradient_to_vector(self):
        grads = np.array([
            [-0.37347779],
            [-1.47903216],
            [0.17596143],
            [-1.33685036],
            [-0.01967514],
            [-0.08573553],
            [0.01188465],
            [-0.07674312],
            [0.03916037],
            [-0.05539735],
            [0.04872715],
            [-0.09359393],
            [-0.05337778],
            [-0.21138458],
            [0.02514856],
            [-0.19106384],
            [0.],
            [0.],
            [0.],
            [0.],
            [0.63290787],
            [0.0372514],
            [-0.06401301],
            [0.09045575],
            [0.],
            [0.],
            [0.],
            [0.],
            [0.],
            [0.],
            [0.],
            [0.],
            [0.],
            [0.],
            [0.],
            [0.91580165],
            [0.02451548],
            [-0.10797954],
            [0.90281891],
            [0.],
            [0.],
            [0.],
            [0.19763343],
            [0.],
            [0.],
            [2.24404238],
            [0.21225753]])
        keys = ['dW1'] * 20 + ['db1'] * 5 \
            +  ['dW2'] * 15 + ['db2'] * 3 \
            +  ['dW3'] * 3  + ['db3'] * 1

        output = {
            'keys': keys,
            'grads': grads
        }
        return output

    def test_gradient_to_vector(self):
        output = self.data_gradient_to_vector()
        grads, keys = gc.gradient_to_vector(self.grads, self.layer_dims)

        assert_almost_equal(grads, output['grads'])
        self.assertListEqual(keys, output['keys'])
        