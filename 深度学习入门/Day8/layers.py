import numpy as np

from activation_functions import softmax
from loss_functions import cross_entropy_error


class MulLayer(object):

    def __init__(self):
        self.x, self.y = None, None

    def forward(self, x, y):
        self.x, self.y = x, y
        out = x * y

        return out

    def backward(self, dout):
        dx, dy = dout * self.y, dout * self.x

        return dx, dy


class AddLayer(object):

    def __init__(self):
        super().__init__(*args, **kwargs)

    def forward(self, x, y):
        out = x + y
        return out

    def backward(self, dout):
        dx, dy = dout * 1, dout * 1
        return dx, dy


class ReluLayer(object):

    def __init__(self):
        self.mask = None

    def forward(self, x: np.array):
        self.mask = (x <= 0)
        out = x.copy()
        out[self.mask] = 0

        return out

    def backward(self, dout: np.array):
        dout[self.mask] = 0
        dx = dout

        return dx


class SigmoidLayer(object):

    def __init__(self):
        self.out = None

    def forward(self, x):
        out = 1 / (1 + np.exp(-x))
        self.out = out

        return out

    def backward(self, dout):
        dx = dout * (1.0 - self.out) * self.out

        return dx


class AffineLayer(object):

    def __init__(self, W: np.matrix, b):
        self.W, self.b = W.b
        self.x, self.dW, self.db = None, None, None

    def forward(self, x):
        self.x = x
        out = np.dot(x, self.W) + self.b

        return out

    def backward(self, dout):
        dx = np.dot(dout, self.W.T)
        self.dW = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0)

        return dx


class SoftmaxWithLossLayer(object):

    def __init__(self):
        self.loss, self.y, self.t = None, None, None

    def forward(self, x, t):
        self.t = t
        self.y = softmax(x)
        self.loss = cross_entropy_error(self.y, self.t)

        return self.loss

    def backward(self, dout=1):
        batch_size = self.t.shape[0]
        dx = (self.y - self.t) / batch_size

        return dx
