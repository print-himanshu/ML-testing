from unittest import TestCase
import numpy as np
from numpy.testing import assert_almost_equal

import logistic_regression


class TestLogisticRegresssion(TestCase):
    def setUp(self):
        self.theta = np.array([[2],
                               [1.],
                               [2.]])

        self.x = np.array([[1, 1, 1], [1., 2., -1.], [3., 4., -3.2]])
        self.y = np.array([[1, 0, 1]])

    def tearDown(self):
        self.theta = 0
        self.x = 0
        self.y = 0

    def testSigmoid(self):
        result = logistic_regression.sigmoid(np.array([0, 2]))
        answer = np.array([0.5, 0.88079708])

        assert_almost_equal(result, answer)

    def testCost(self):
        a = logistic_regression.sigmoid(logistic_regression.z(self.theta, self.x))
        result = logistic_regression.J(self.y, a)
        self.assertAlmostEqual(np.squeeze(result), 5.80154532, places=4)

    def test_Z(self):
        result = logistic_regression.z(self.theta, self.x)
        answer = np.array([[9., 12., -5.4]])
        assert_almost_equal(result, answer)

    def test_sigmoid_2(self):
        z = logistic_regression.z(self.theta, self.x)
        result = logistic_regression.sigmoid(z)
        answer = np.array([[0.99987661, 0.99999386, 0.00449627]])

        assert_almost_equal(result, answer)

    def test_gradient_differentiation(self):
        z = logistic_regression.z(self.theta, self.x)
        a = logistic_regression.sigmoid(z)
        result = logistic_regression.gradient_differentiation(a, self.y, self.x)

        answer = np.array([[0.00145557813678], [0.99845601], [2.39507239]])

        assert_almost_equal(result, answer)

    def test_forward_propagation(self):
        theta, costs = logistic_regression.forward_propagation(
            self.theta, self.x, self.y, 0.009, 100)
        answer = np.array([[1.92535983],
                           [0.19033591],
                           [0.12259159]])

        assert_almost_equal(theta, answer)

    def test_prediction(self):
        x = np.array([[1, 1, 1], [1., -1.1, -3.2], [1.2, 2., 0.1]])
        theta = np.array([[-0.3], [0.1124579], [0.23106775]])

        result = logistic_regression.predict(theta, x)
        answer = np.array([[1.,  1., 0.]])

        assert_almost_equal(result, answer)
