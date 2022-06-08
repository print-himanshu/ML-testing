from unittest import TestCase
import nn
import numpy as np
from numpy.testing import assert_almost_equal


class TestShallowNN(TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_layer_size(self):
        X = np.zeros((3, 5))
        Y = np.zeros((1, 5))
        result = nn.layer_size(X, Y)
        answer = (3, 4, 1)

        self.assertEqual(result, answer)

    def test_initialize_parameter_shape(self):
        result = nn.initilaize_parameter(2, 4, 1)

        self.assertEqual(result['W1'].shape, (4, 2))
        self.assertEqual(result['b1'].shape, (4, 1))
        self.assertEqual(result['W2'].shape, (1, 4))
        self.assertEqual(result['b2'].shape, (1, 1))

    def test_initialize_paramter_value(self):
        result = nn.initilaize_parameter(2, 4, 1)

        W1 = np.array([[-0.00416758, -0.00056267],
                       [-0.02136196,  0.01640271],
                       [-0.01793436, -0.00841747],
                       [0.00502881, -0.01245288]])
        b1 = np.array([[0.],
                       [0.],
                       [0.],
                       [0.]])

        W2 = np.array([[-0.01057952, -0.00909008,  0.00551454,  0.02292208]])
        b2 = np.array([[0.]])

        assert_almost_equal(result['W1'], W1)
        assert_almost_equal(result['W2'], W2)
        assert_almost_equal(result['b1'], b1)
        assert_almost_equal(result['b2'], b2)

    def test_sigmoid(self):
        result = nn.sigmoid(np.array([0, 2]))
        answer = np.array([0.5, 0.88079708])

        assert_almost_equal(answer, result)

    def test_forward_propagation(self):
        np.random.seed(1)
        X_assess = np.random.randn(2, 3)
        b1 = np.random.randn(4, 1)
        b2 = np.array([[-1.3]])
        W1 = np.array([[-0.00416758, -0.00056267],
                       [-0.02136196,  0.01640271],
                       [-0.01793436, -0.00841747],
                       [0.00502881, -0.01245288]])

        W2 = np.array([[-0.01057952, -0.00909008,  0.00551454,  0.02292208]])

        parameter = {'W1': W1,
                     'b1': b1,
                     "W2": W2,
                     "b2": b2}

        a2, answer = nn.forward_propagation(X_assess, parameter)

        self.assertAlmostEqual(np.mean(answer['z1']), 0.26281864019752443)
        self.assertAlmostEqual(np.mean(answer['a1']), 0.09199904522700109)
        self.assertAlmostEqual(np.mean(answer['z2']), -1.3076660128732143)
        self.assertAlmostEqual(np.mean(answer['a2']), 0.21287768171914198)

    def test_cost(self):
        np.random.seed(1)
        Y_assess = np.random.randn(1, 3) > 0

        a2 = np.array([[0.5002307,  0.49985831,  0.50023963]])

        output = nn.compute_cost(a2, Y_assess)
        answer = 0.6930587610394646

        self.assertAlmostEqual(answer, output)

    def backward_propagation_test_case(self):
        np.random.seed(1)
        X_assess = np.random.randn(2, 3)
        Y_assess = (np.random.randn(1, 3) > 0)
        parameters = {'W1': np.array([[-0.00416758, -0.00056267],
                                      [-0.02136196,  0.01640271],
                                      [-0.01793436, -0.00841747],
                                      [0.00502881, -0.01245288]]),
                      'W2': np.array([[-0.01057952, -0.00909008,  0.00551454,  0.02292208]]),
                      'b1': np.array([[0.],
                                      [0.],
                                      [0.],
                                      [0.]]),
                      'b2': np.array([[0.]])}

        cache = {'a1': np.array([[-0.00616578,  0.0020626,  0.00349619],
                                 [-0.05225116,  0.02725659, -0.02646251],
                                 [-0.02009721,  0.0036869,  0.02883756],
                                 [0.02152675, -0.01385234,  0.02599885]]),
                 'a2': np.array([[0.5002307,  0.49985831,  0.50023963]]),
                 'z1': np.array([[-0.00616586,  0.0020626,  0.0034962],
                                 [-0.05229879,  0.02726335, -0.02646869],
                                 [-0.02009991,  0.00368692,  0.02884556],
                                 [0.02153007, -0.01385322,  0.02600471]]),
                 'z2': np.array([[0.00092281, -0.00056678,  0.00095853]])}

        dw1 = np.array([[0.00301023, -0.00747267], [0.00257968, -0.00641288],
                       [-0.00156892, 0.003893], [-0.00652037, 0.01618243]])
        db1 = np.array([[0.00176201], [0.00150995],
                       [-0.00091736], [-0.00381422]])
        dw2 = np.array([[0.00078841, 0.01765429, -0.00084166, -0.01022527]])
        db2 = np.array([[-0.16655712]])

        result = {'dw1': dw1,
                  'db1': db1,
                  'dw2': dw2,
                  'db2': db2}

        return X_assess, Y_assess, parameters, cache, result

    def test_backward_propagation(self):
        X_assess, Y_assess, parameters, cache, result = self.backward_propagation_test_case()

        output = nn.backward_propagation(X_assess, Y_assess, cache, parameters)

        assert_almost_equal(output['dw2'], result['dw2'])
        assert_almost_equal(output['db2'], result['db2'])
        assert_almost_equal(output['dw1'], result['dw1'])
        assert_almost_equal(output['db1'], result['db1'])

    def update_parameters_test_case(self):
        parameters = {'W1': np.array([[-0.00615039,  0.0169021],
                                      [-0.02311792,  0.03137121],
                                      [-0.0169217, -0.01752545],
                                      [0.00935436, -0.05018221]]),
                      'W2': np.array([[-0.0104319, -0.04019007,  0.01607211,  0.04440255]]),
                      'b1': np.array([[-8.97523455e-07],
                                      [8.15562092e-06],
                                      [6.04810633e-07],
                                      [-2.54560700e-06]]),
                      'b2': np.array([[9.14954378e-05]])}

        grads = {'dw1': np.array([[0.00023322, -0.00205423],
                                  [0.00082222, -0.00700776],
                                  [-0.00031831,  0.0028636],
                                  [-0.00092857,  0.00809933]]),
                 'dw2': np.array([[-1.75740039e-05,   3.70231337e-03,  -1.25683095e-03,
                                   -2.55715317e-03]]),
                 'db1': np.array([[1.05570087e-07],
                                  [-3.81814487e-06],
                                  [-1.90155145e-07],
                                  [5.46467802e-07]]),
                 'db2': np.array([[-1.08923140e-05]])}

        W1 = np.array([[-0.00643025, 0.01936718], [-0.02410458, 0.03978052],
                      [-0.01653973, -0.02096177], [0.01046864, -0.05990141]])

        b1 = np.array([[-1.02420756e-06], [1.27373948e-05],
                      [8.32996807e-07], [-3.20136836e-06]])

        W2 = np.array([[-0.01041081, -0.04463285, 0.01758031, 0.04747113]])
        b2 = np.array([[0.00010457]])

        result = {'W1': W1,
                  'b1': b1,
                  'W2': W2,
                  'b2': b2}

        return parameters, grads, result

    def test_update_parameters(self):
        parameters, grads, result = self.update_parameters_test_case()
        output = nn.update_parameters(parameters, grads, 1.2)

        assert_almost_equal(output['W1'], result['W1'])
        assert_almost_equal(output['b1'], result['b1'])
        assert_almost_equal(output['W2'], result['W2'])
        assert_almost_equal(output['b2'], result['b2'])

    def nn_model_test_case(self):
        np.random.seed(1)
        X_assess = np.random.randn(2, 3)
        Y_assess = (np.random.randn(1, 3) > 0)

        W1 = np.array([[-0.65848169, 1.21866811], [-0.76204273, 1.39377573],
                      [0.5792005, -1.10397703], [0.76773391, -1.41477129]])
        b1 = np.array([[0.287592], [0.3511264], [-0.2431246], [-0.35772805]])
        W2 = np.array([[-2.45566237, -3.27042274, 2.00784958, 3.36773273]])
        b2 = np.array([[0.20459656]])

        result = {'W1': W1, 
                  'b1': b1,
                  "W2": W2,
                  "b2": b2}

        return X_assess, Y_assess, result

    def test_nn_model(self):
        X, Y , result = self.nn_model_test_case()

        output, _ = nn.nn_model(X, Y , 1.2, num_iteration = 10000, print_cost = True)

        assert_almost_equal(output['W1'], result['W1'])
        assert_almost_equal(output['b1'], result['b1'])
        assert_almost_equal(output['W2'], result['W2'])
        assert_almost_equal(output['b2'], result['b2'])
        
