import optimization as op
import numpy as np
from numpy.testing import assert_almost_equal
from unittest import TestCase
import numpy as np
import os


class TestOptimization(TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    # ----------------------- Mini batch random -------------------
    def data_mini_batch_random(self):
        np.random.seed(1)
        mini_batch_size = 64
        X = np.random.randn(12288, 148)
        y = (np.random.randn(1, 148) < 0.5).astype(int)
        input = [
            X,
            y,
            mini_batch_size
        ]
        file_path = os.path.join(
            os.getcwd(), "optimization_algorithm", "output", "mini_batch.npy")
        output = np.load(file_path, allow_pickle=True)

        return input, output

    def test_mini_batch_random(self):
        input, output = self.data_mini_batch_random()
        result = op.mini_batch_random(*input)

        for output_batch, result_batch in zip(output, result):
            assert_almost_equal(output_batch[0], result_batch[0])
            assert_almost_equal(output_batch[1], result_batch[1])

    # ----------------------- Initialize Velocity -----------------
    def data_initialize_velocity(self):
        np.random.seed(1)
        W1 = np.random.randn(2, 3)
        b1 = np.random.randn(2, 1)
        W2 = np.random.randn(3, 3)
        b2 = np.random.randn(3, 1)
        parameters = {
            "W1": W1, "b1": b1,
            "W2": W2, "b2": b2,
        }

        input = parameters

        v = {'dW1': np.array([
            [0., 0., 0.],
            [0., 0., 0.]]),
            'db1': np.array([
                [0.],
                [0.]]),
            'dW2': np.array([[
                0., 0., 0.],
                [0., 0., 0.],
                [0., 0., 0.]]),
            'db2': np.array([
                [0.],
                [0.],
                [0.]])}

        output = v

        return input, output

    def test_initialize_velocity(self):
        input, output = self.data_initialize_velocity()
        result = op.initialize_velocity(input)

        for key, value in result.items():
            assert_almost_equal(output[key], value)

    # ----------------------- Update parameter (Momentum) ---------
    def data_update_parameter_with_velocity(self):
        np.random.seed(1)
        W1 = np.random.randn(2, 3)
        b1 = np.random.randn(2, 1)
        W2 = np.random.randn(3, 3)
        b2 = np.random.randn(3, 1)

        dW1 = np.random.randn(2, 3)
        db1 = np.random.randn(2, 1)
        dW2 = np.random.randn(3, 3)
        db2 = np.random.randn(3, 1)

        parameters = {
            "W1": W1, "b1": b1, "W2": W2, "b2": b2}
        grads = {
            "dW1": dW1, "db1": db1, "dW2": dW2, "db2": db2}
        v = {
            'dW1': np.array([[0.,  0.,  0.],
                             [0.,  0.,  0.]]), 'dW2': np.array([[0.,  0.,  0.],
                                                                [0.,  0.,  0.],
                                                                [0.,  0.,  0.]]), 'db1': np.array([[0.],
                                                                                                   [0.]]), 'db2': np.array([[0.],
                                                                                                                            [0.],
                                                                                                                            [0.]])}

        input = [
            parameters,
            grads,
            v,
            0.9,
            0.01
        ]

        output_parameters = {
            'W1': np.array([
                [1.62544598, -0.61290114, -0.52907334],
                [-1.07347112,  0.86450677, -2.30085497]]),
            'b1': np.array([
                [1.74493465],
                [-0.76027113]]),
            'W2': np.array([
                [0.31930698, -0.24990073,  1.4627996],
                [-2.05974396, -0.32173003, -0.38320915],
                [1.13444069, -1.0998786, -0.1713109]]),
            'b2': np.array([
                [-0.87809283],
                [0.04055394],
                [0.58207317]])}

        v = {
            'dW1': np.array([
                [-0.11006192,  0.11447237,  0.09015907],
                [0.05024943,  0.09008559, -0.06837279]]),
            'dW2': np.array([
                [-0.02678881,  0.05303555, -0.06916608],
                [-0.03967535, -0.06871727, -0.08452056],
                [-0.06712461, -0.00126646, -0.11173103]]),
            'db1': np.array([
                [-0.01228902],
                [-0.09357694]]),
            'db2': np.array([
                [0.02344157],
                [0.16598022],
                [0.07420442]])}

        output = {
            'parameters': output_parameters,
            'v': v
        }

        return input, output

    def test_update_parameters_with_velocity(self):
        input, output = self.data_update_parameter_with_velocity()
        parameters, v = op.update_parameters_with_velocity(*input)

        for key, value in parameters.items():
            assert_almost_equal(output['parameters'][key], value)

        for key, value in v.items():
            assert_almost_equal(output['v'][key], value)

    # ----------------------- Initialize Adam ---------------------
    def data_initialize_adam(self):
        np.random.seed(1)
        W1 = np.random.randn(2, 3)
        b1 = np.random.randn(2, 1)
        W2 = np.random.randn(3, 3)
        b2 = np.random.randn(3, 1)

        parameters = {
            'W1': W1, 'b1': b1,
            "W2": W2, "b2": b2
        }
        input = parameters
        v = {
            'dW1': np.array([
                [0., 0., 0.],
                [0., 0., 0.]]),
            'db1': np.array([
                [0.],
                [0.]]),
            'dW2': np.array([
                [0., 0., 0.],
                [0., 0., 0.],
                [0., 0., 0.]]),
            'db2': np.array([
                [0.],
                [0.],
                [0.]])}

        s = {
            'dW1': np.array([
                [0., 0., 0.],
                [0., 0., 0.]]),
            'db1': np.array([
                [0.],
                [0.]]),
            'dW2': np.array([
                [0., 0., 0.],
                [0., 0., 0.],
                [0., 0., 0.]]),
            'db2': np.array([
                [0.],
                [0.],
                [0.]])}

        output = {
            'v': v,
            's': s
        }

        return input, output

    def test_initialize_adam(self):
        input, output = self.data_initialize_adam()
        v, s = op.initialize_adam(input)

        for key, value in v.items():
            assert_almost_equal(output['v'][key], value)

        for key, value in s.items():
            assert_almost_equal(output['s'][key], value)

    # ----------------------- Update Parameters (Adam) ------------
    def data_update_parametes_with_adam(self):
        v, s = (
            {
                'dW1': np.array([
                    [0.,  0.,  0.],
                    [0.,  0.,  0.]]),
                'dW2': np.array([
                    [0.,  0.,  0.],
                    [0.,  0.,  0.],
                    [0.,  0.,  0.]]),
                'db1': np.array([
                    [0.],
                    [0.]]),
                'db2': np.array([
                    [0.],
                    [0.],
                    [0.]])
            },
            {
                'dW1': np.array([
                    [0.,  0.,  0.],
                    [0.,  0.,  0.]]),
                'dW2': np.array([
                    [0.,  0.,  0.],
                    [0.,  0.,  0.],
                    [0.,  0.,  0.]]),
                'db1': np.array([
                    [0.],
                    [0.]]),
                'db2': np.array([
                    [0.],
                    [0.],
                    [0.]])
            }
        )
        np.random.seed(1)
        W1 = np.random.randn(2, 3)
        b1 = np.random.randn(2, 1)
        W2 = np.random.randn(3, 3)
        b2 = np.random.randn(3, 1)

        dW1 = np.random.randn(2, 3)
        db1 = np.random.randn(2, 1)
        dW2 = np.random.randn(3, 3)
        db2 = np.random.randn(3, 1)

        parameters = {"W1": W1, "b1": b1, "W2": W2, "b2": b2}
        grads = {"dW1": dW1, "db1": db1, "dW2": dW2, "db2": db2}

        input = [
            parameters,
            grads,
            v,
            s,
            2,
            0.01,
            0.9,
            0.999,
            1e-8
        ]
        v = {
            'dW1': np.array([
                [-0.11006192,  0.11447237,  0.09015907],
                [0.05024943,  0.09008559, -0.06837279]]),
            'dW2': np.array([
                [-0.02678881,  0.05303555, -0.06916608],
                [-0.03967535, -0.06871727, -0.08452056],
                [-0.06712461, -0.00126646, -0.11173103]]),
            'db1': np.array([
                [-0.01228902],
                [-0.09357694]]),
            'db2': np.array([
                [0.02344157],
                [0.16598022],
                [0.07420442]])}

        s = {
            'dW1': np.array([
                [0.00121136, 0.00131039, 0.00081287],
                [0.0002525, 0.00081154, 0.00046748]]),
            'dW2': np.array([
                [7.17640232e-05, 2.81276921e-04, 4.78394595e-04],
                [1.57413361e-04, 4.72206320e-04, 7.14372576e-04],
                [4.50571368e-04, 1.60392066e-07, 1.24838242e-03]]),
            'db1': np.array([
                [1.51020075e-05],
                [8.75664434e-04]]),
            'db2': np.array([
                [5.49507194e-05],
                [2.75494327e-03],
                [5.50629536e-04]])}

        output_parameters = {
            'W1': np.array([
                [1.63178673, -0.61919778, -0.53561312],
                [-1.08040999,  0.85796626, -2.29409733]]),
            'b1': np.array([
                [1.75225313],
                [-0.75376553]]),
            'W2': np.array([
                [0.32648046, -0.25681174,  1.46954931],
                [-2.05269934, -0.31497584, -0.37661299],
                [1.14121081, -1.09244991, -0.16498684]]),
            'b2': np.array([
                [-0.88529979],
                [0.03477238],
                [0.57537385]])}

        output = {
            'v': v,
            's': s,
            'parameters': output_parameters
        }

        return input, output

    def test_update_parameters_with_adam(self):
        input, output = self.data_update_parametes_with_adam()
        parameters, v, s = op.update_parameters_with_adam(*input)

        for key, value in v.items():
            assert_almost_equal(output['v'][key], value)

        for key, value in s.items():
            assert_almost_equal(output['s'][key], value)

        for key, value in parameters.items():
            assert_almost_equal(output['parameters'][key], value)


if __name__ == "__main__":
    pass
