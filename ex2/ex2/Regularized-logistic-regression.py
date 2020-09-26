import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as optimize


def feature_map(X):
    m = len(X)
    x_1 = X[:, 0][np.newaxis].T
    x_2 = X[:, 1][np.newaxis].T
    X = np.concatenate(
        (
            np.ones((m, 1)),
            x_1,
            x_2,
            x_1**2,
            x_1 * x_2,
            x_2**2,
            x_1**3,
            x_1**2 * x_2,
            x_1 * x_2**2,
            x_2**3,
            x_1**4,
            x_1**3 * x_2,
            x_1**2 * x_2**2,
            x_1**1 * x_2**3,
            x_2**4,
            x_1**5,
            x_1**4 * x_2,
            x_1**3 * x_2**2,
            x_1**2 * x_2**3,
            x_1**1 * x_2**4,
            x_2**5,
            x_1**6,
            x_1**5 * x_2,
            x_1**4 * x_2**2,
            x_1**3 * x_2**3,
            x_1**2 * x_2**4,
            x_1 * x_2**5,
            x_2**6
        ),axis=1)
    return X


def sigmoid(z):
    a = 1 + np.exp(-z)
    return 1 / a


def cost_function(theta, X, y, _lambda):
    m = len(X)
    h_theta = sigmoid(np.dot(X, theta[np.newaxis].T))

    a = - np.dot(y.T, np.log(h_theta))
    b = - np.dot((1 - y).T, np.log(1 - h_theta))
    c = a + b
    d = _lambda * np.dot(theta[1:], theta[1:].T) / m / 2 # only if j >= 1
    J_val = c / m + d

    return J_val[0][0]


def gradient(theta, X, y, _lambda):
    m = len(X)
    h_theta = sigmoid(np.dot(X, theta[np.newaxis].T))

    # Calculate gradient descent
    a = h_theta - y
    b = _lambda * theta / m
    gradience = (np.dot(X.T, a) / m).ravel()
    gradience[1:] = gradience[1:] + b[:-1] # theta[j]: only if j >= 1

    return gradience


def predict(x, theta):
    '''
    x = array(m, n)
    theta = array(n,)
    '''
    pred = np.zeros((len(x), 1))
    a = sigmoid(np.dot(x, theta[np.newaxis].T))
    pred[sigmoid(np.dot(x, theta[np.newaxis].T)) >= .5] = 1
    return pred    


def main():
    data = np.loadtxt('./ex2/ex2/ex2data2.txt', delimiter=',')
    X = data[:, :-1]
    y = data[:, -1][np.newaxis].T

    

    X = feature_map(X)
    theta = np.zeros((28,), dtype=np.float)
    _lambda = 0.1
    
    res = optimize.minimize(cost_function, theta, (X, y, _lambda), 'BFGS', gradient)
    theta = res.x
    
    # Plot decision boundary
    _x, _y = np.meshgrid(np.arange(-1, 1.5, 0.01), np.arange(-1, 1.5, 0.01))
    _X = np.c_[_x.reshape(-1, 1), _y.reshape(-1, 1)]
    _X = feature_map(_X)
    z = predict(_X, theta)
    z = z.reshape(_x.shape)
    plt.contour(_x, _y, z, cmap=plt.cm.Spectral)

    # Plot data
    plt.scatter(X[:, 1], X[:, 2], c = y.ravel(), cmap=plt.cm.Spectral)
    
    plt.xlabel('Microchip test 1')
    plt.ylabel('Microchip test 2')
    plt.show()


if __name__ == '__main__':
    main()