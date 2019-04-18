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
        self.b = 0.7

        return 1 if self.neuron_input() > 0 else 0

    def OR(self):
        self.w = np.array([0.5, 0.5])
        self.b = -0.2

        return 1 if self.neuron_input() > 0 else 0


class Multi_Layered_Perceptron(Perceptron):

    def __init__(self, x1, x2):
        super().__init__(x1, x2)

    def XOR(self):
        self.s1 = self.NAND()
        self.s2 = self.OR()

        return Perceptron(self.s1, self.s2).AND()


for x1, x2 in [(0, 0), (1, 0), (0, 1), (1, 1)]:

    print(x1, x2, Multi_Layered_Perceptron(x1, x2).XOR())
