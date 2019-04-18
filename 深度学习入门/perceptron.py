import numpy as np


class Perceptron(object):

    def __init__(self, x1, x2):
        self.x1, self.x2 = x1, x2
        self.x = np.array([self.x1, self.x2])

    def input_signal(self):
        return np.sum(self.w * self.x) + self.b

    def activation(self):
        def step_function(x): return 1 if x > 0 else 0
        return step_function(self.input_signal())

    def AND(self):
        self.w = np.array([0.5, 0.5])
        self.b = -0.7

        return self.activation()

    def NAND(self):
        self.w = np.array([-0.5, -0.5])
        self.b = 0.7

        return self.activation()

    def OR(self):
        self.w = np.array([0.5, 0.5])
        self.b = -0.2

        return self.activation()


class Multi_Layered_Perceptron(Perceptron):

    def __init__(self, x1, x2):
        super().__init__(x1, x2)

    def XOR(self):
        self.s1 = self.NAND()
        self.s2 = self.OR()

        return Perceptron(self.s1, self.s2).AND()


for x1, x2 in [(0, 0), (1, 0), (0, 1), (1, 1)]:

    print(x1, x2, Multi_Layered_Perceptron(x1, x2).XOR())
