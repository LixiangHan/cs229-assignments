import numpy as np
from scipy import io
from matplotlib import pyplot as plt
from scipy.optimize import optimize


class Network():
    def __init__(self):
        weights = io.loadmat('./ex3/ex3/ex3weights.mat')
        self.Theta1 = weights['Theta1']
        self.Theta2 = weights['Theta2']

    def forward(self, X):
        X = X.T
        a_1 = np.vstack((np.ones((1, X.shape[1])), X))
        z_2 = np.dot(self.Theta1, a_1)
        a_2 = sigmoid(z_2)
        a_2 = np.vstack((np.ones((1, a_2.shape[1])), a_2))
        z_3 = np.dot(self.Theta2, a_2)
        a_3 = sigmoid(z_3)
        return a_3


def sigmoid(z):
    a = 1 + np.exp(-z)
    return 1 / a


def main():
    data = io.loadmat('./ex3/ex3/ex3data1.mat')
    X = data['X']
    y = data['y']
    net = Network()
    acc = np.where((np.argmax(net.forward(X), axis=0) + 1) == y.T)[0].size / 5000.
    print('The accuracy is: %.4f.' % acc)


if __name__ == '__main__':
    main()
