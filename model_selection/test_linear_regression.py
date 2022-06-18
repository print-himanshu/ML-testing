import linear_regression_poly as lr
import numpy as np
from unittest import TestCase
from numpy.testing import assert_almost_equal
import pandas as pd
import os


class TestDegreeSelect(TestCase):
    def setUp(self):
        self.train_x = np.array([
            [-0.23634686,   0.113534535, -0.101701414,  0.063736181, -0.085990661,  0.17726076]])
        self.train_y = np.array([
            [15, 18, 19, 15, 10, 10]])

        self.dev_x = np.array([[-0.111036305,  0.018121427]])
        self.dev_y = np.array([[11, 17]])

        self.test_x = np.array([[0.056434487, -0.056651023]])
        self.test_y = np.array([[16, 19]])
        self.z = np.array([
            [-0.415576005, 0.175594258, -0.171097089, 0.100890736, -0.143929991, 0.264798281,
             ]])

    def tearDown(self):
        pass

    # ------------------------- Polynomail Feature ---------------------
    def data_poly_feature(self):
        input = [
            self.train_x,
            10
        ]

        path = os.path.join(os.getcwd(), 'model_selection',
                            'ouput', 'x_degree.xlsx')
        output = pd.read_excel(path)

        output = output.to_numpy()
        output = output.T

        return input, output

    def test_poly_feature(self):
        input, output = self.data_poly_feature()
        result = lr.polynomial_feature(*input)

        assert_almost_equal(output, result)

    # ------------------------- Weight Initializtion -------------------
    def data_weight_initialization(self):
        input = 10

        path = os.path.join(os.getcwd(), 'model_selection',
                            'ouput', 'weight.xlsx')
        output = pd.read_excel(path)
        output = output.to_numpy()

        return input, output

    def test_weight_initialization(self):
        input, output = self.data_weight_initialization()
        W, b = lr.weight_initialization(input)

        assert_almost_equal(output, W)

    # ------------------------- Hypothesis -----------------------------
    def data_hypotesis(self):
        degree = 10
        W, b = lr.weight_initialization(degree)
        x_poly = lr.polynomial_feature(self.train_x, degree)

        input = [
            x_poly,
            W,
            b
        ]

        output = self.z

        return input, output

    def test_hypothesis(self):
        input, output = self.data_hypotesis()
        result = lr.hypothesis(*input)

        assert_almost_equal(result, output)

    # ------------------------- MSE cost -------------------------------
    def data_mse_cost(self):
        input = [
            self.z,
            self.train_y
        ]

        output = 111.8781094

        return input, output

    def test_mse_cost(self):
        input, output = self.data_mse_cost()
        result = lr.mse_cost(*input)

        self.assertAlmostEqual(output, result)

    # ------------------------- Gradient Descent -----------------------

    def data_gradient_descent(self):
        degree = 10
        x_poly = lr.polynomial_feature(self.train_x, degree)

        input = [
            self.z,
            self.train_y,
            x_poly
        ]

        dW = np.array([
            [0.294411114, -0.28843103, 0.024328566, -0.010587702, 0.001594876, -0.000508907, 9.66048E-05, -2.67213E-05, 5.62675E-06, -1.44853E-06
             ]])

        db = np.array([[-14.5315533]])

        output = {
            'dW': dW,
            'db': db
        }
        return input, output

    def test_gradient_descent(self):
        input, output = self.data_gradient_descent()
        dW, db = lr.gradient_descent(*input)

        assert_almost_equal(dW, output['dW'])
        assert_almost_equal(db, output['db'])

    # ------------------------- Update Parameters ----------------------
    def data_update_parameters(self):
        degree = 10
        W, b = lr.weight_initialization(10)
        x_poly = lr.polynomial_feature(self.train_x, degree)
        z = lr.hypothesis(x_poly, W, b)
        dW, db = lr.gradient_descent(
            z,
            self.train_y,
            x_poly
        )

        input = [
            W, b,
            dW, db,
            0.1
        ]

        W = np.array([
            [1.594904252, -0.582913307, -0.530604607, -1.07190985, 0.865248142, -2.301487809, 1.7448021, -0.761204228, 0.319038537, -0.249370235
             ]])

        b = np.array([[1.45315533]])

        output = {
            'W': W,
            'b': b
        }
        return input, output

    def test_update_parameter(self):
        input, output = self.data_update_parameters()
        W, b = lr.update_parameter(*input)

        assert_almost_equal(W, output['W'])
        assert_almost_equal(b, output['b'])