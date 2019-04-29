from collections import OrderedDict

import numpy as np

from layers import *


def numerical_gradient(f, x):
    h = 1e-4
    grad = np.zeros_like(x)

    for idx in range(x.size):
        tmp = x[idx]

        # f(x+h)
        x[idx] = tmp + h
        fxh1 = f(x)

        # f(x-h)
        x[idx] = tmp - h
        fxh2 = f(x)

        grad[idx] = (fxh1 - fxh2) / (2 * h)
        x[idx] = tmp

    return grad


class TwoLayerNet(object):

    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):

        self.params = {}
        self.params['W1'] = weight_init_std * \
            np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = weight_init_std * \
            np.random.randn(hidden_size, output_size)

        self.layers = OrderedDict()
