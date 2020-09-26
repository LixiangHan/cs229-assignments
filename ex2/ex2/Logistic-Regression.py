import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as optimize


def sigmoid(z):
    a = 1 + np.exp(-z)
    return 1 / a


def cost_function(theta, X, y):
    m = len(X)
    X = np.concatenate((np.ones((m, 1)), X), axis=1)
    # Calculate Cost J
    h_theta = sigmoid(np.dot(X, theta[np.newaxis].T))

    a = - np.dot(y, np.log(h_theta))
    b = - np.dot((1 - y), np.log(1 - h_theta))
    c = a + b
    J_val = c / m

    return J_val[0]


def gradient(theta, X, y):
    m = len(X)
    X = np.concatenate((np.ones((m, 1)), X), axis=1)
    h_theta = sigmoid(np.dot(X, theta[np.newaxis].T))

    # Calculate gradient descent
    a = h_theta - y[np.newaxis].T
    gradience = np.dot(X.T, a) / m

    return gradience.T[0]

def main():
    data = np.loadtxt('./ex2/ex2/ex2data1.txt', delimiter=',')
    X = data[:, :-1]
    y = data[:, -1]
    theta = np.zeros((3, ))
    
    # Plot data
    plot_1 = plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], marker='*')
    plot_2 = plt.scatter(X[y == 0][:, 0], X[y == 0][:, 1], marker='x')
    plt.xlabel('Exam 1 score')
    plt.ylabel('Exam 2 score')
    plt.legend([plot_1, plot_2], ['Admitted', 'Not admitted'], loc='upper right')
    

    res = optimize.minimize(cost_function, theta, (X, y), 'BFGS', gradient)
    theta = res.x
    
    x_1 = np.linspace(20, 100, 1000)
    x_2 = - (theta[0] + theta[1] * x_1) / theta[2]
    plt.plot(x_1, x_2, c='red')
    plt.show()
    

if __name__ == '__main__':
    main()
    