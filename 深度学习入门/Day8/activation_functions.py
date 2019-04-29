import numpy as np


def step_function(x: np.array):
    y: np.array = x > 0
    return y.astype(np.int)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def relu(x):
    return np.maximum(0, x)


def identity_function(x):
    return x


def softmax(a: np.array):
    c = np.max(a)
    exp_a = np.exp(a - c)  # to avoid overflow
    sum_exp_a = np.sum(exp_a)
    y: np.array = exp_a / sum_exp_a

    return y
