import numpy as np


class Perceptron(object):

    def __init__(self, x1, x2):
        self.x1, self.x2 = x1, x2
        self.x = np.array([self.x1, self.x2])

    def neuron_input(self):
        return np.sum(self.w * self.x) + self.b

    def AND(self):
        self.w = np.array([0.5, 0.5])
        self.b = -0.7

        return 1 if self.neuron_input() > 0 else 0

    def NAND(self):
        self.w = np.array([-0.5, -0.5])
        self.b = -0.7

        return 1 if self.neuron_input() > 0 else 0

    def OR(self):
        self.w = np.array([0.5, 0.5])
        self.b = -0.2

        return 1 if self.neuron_input() > 0 else 0


p = Perceptron(1, 2)
print([p.AND(), p.NAND(), p.OR()])
