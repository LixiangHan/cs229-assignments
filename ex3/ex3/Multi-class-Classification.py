import numpy as np
from scipy import io
from matplotlib import pyplot as plt
import scipy.optimize as optimize
import random


class Classifier():
    def __init__(self, X, y, _lambda):
        self.X = X
        self.y = y
        self._lambda = _lambda
        self.theta = np.zeros((400,), dtype=np.float)

    def fit(self):
        res = optimize.minimize(cost_function, self.theta,
                                (self.X, self.y, self._lambda), 'BFGS', gradient)
        # print(res)
        self.theta = res.x

    def predict(self, x):
        return h_theta(x, self.theta)


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def h_theta(X, theta):
    return sigmoid(np.dot(X, theta))


def cost_function(theta, X, y, _lambda):
    theta = theta[:][np.newaxis].T
    h = h_theta(X, theta)
    a = np.log(h)
    b = np.log(1 - h)
    c = np.dot(y.T, a)
    d = np.dot((1 - y).T, b)
    e = c + d
    f = _lambda / 2 * np.dot(theta[1:].T, theta[1:])
    m = len(X)
    return ((- e + f) / m)[0][0]


def gradient(theta, X, y, _lambda):
    theta = theta[:][np.newaxis].T
    h = h_theta(X, theta)
    a = h - y
    b = np.dot(X.T, a)
    b[1:] += _lambda * b[1:]
    m = len(X)
    return (b / m).ravel()


def main():
    data = io.loadmat('./ex3/ex3/ex3data1.mat')
    X = data['X']
    y = data['y']
    y[y == 10] = 0  # "0" digit is labeled as "10"

    # randomly show some examples
    # examples = X[random.sample(range(5000), 100)]
    # f, a = plt.subplots(10, 10, figsize=(10, 10))
    # for i in range(10):
    #     for j in range(10):
    #         # have to transpose
    #         a[i][j].imshow(np.reshape(
    #             examples[i * 10 + j], (20, 20)).T, cmap='gray')
    #         a[i][j].set_xticks(())
    #         a[i][j].set_yticks(())
    # plt.show()


    classifiers = []
    for i in range(10):
        temp_y = np.zeros((5000, 1))
        temp_y[y == i] = 1
        classifiers.append(Classifier(X, temp_y, 0.1))
        classifiers[i].fit()

    predictions = []
    for i in range(10):
        predictions.append(classifiers[i].predict(X))
    pred_y = np.vstack(predictions)
    pred_y = np.argmax(pred_y, axis=0)
    
    acc = np.where(y.T[0] == pred_y)[0].size / 5000.
    print('The accuracy is: %.2f.' % acc)
    

if __name__ == '__main__':
    main()
