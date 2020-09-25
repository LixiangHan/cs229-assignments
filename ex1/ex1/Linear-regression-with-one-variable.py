import numpy as np
import matplotlib.pyplot as plt


def cost_function(X, y, theta, m):
    a = np.dot(X, theta) # h(X)
    b = a - y[:][np.newaxis].T
    return np.dot(b.T, b)[0][0] / 2 / m


def gradient_descent(X, y, iteration=1500, alpha=0.01):
    m = len(X) # num of samples
    theta = np.zeros((1, 2)).T # initiate theta = [0, 0]
    X = np.concatenate((np.ones((m, 1)), X[np.newaxis][:].T), axis=1) # add a colum of one to X

    for i in range(iteration):
        theta_temp = theta
        a = np.dot(X, theta_temp) # h(X)
        b = a - y[:][np.newaxis].T
        theta[0] = theta_temp[0] - alpha * np.dot(b.T, X[:,0]) / m
        theta[1] = theta_temp[1] - alpha * np.dot(b.T, X[:,1]) / m

    return theta
        


def main():
    data = np.loadtxt('./ex1/ex1/ex1data1.txt', delimiter=',', )
    X = data[:, 0]
    y = data[:, 1]

    # plot the data
    plt.scatter(X, y, marker='x', c='red')
    plt.xlabel('Population of City in 10,000s')
    plt.ylabel('Profit in $10,000s')

    theta = gradient_descent(X, y)
    print('Profile in areas of 35,000 is: %.2f' % np.dot(np.array([1, 3.5]), theta))
    print('Profile in areas of 70,000 is: %.2f' % np.dot(np.array([1, 7.0]), theta))

    # plot the prediction
    temp_x = np.linspace(5, 25, 20)
    temp_y = theta[0] + theta[1] * temp_x
    plt.plot(temp_x, temp_y)

    plt.legend(['Linear regression', 'Training data'], loc='lower right')
    plt.show()


if __name__ == '__main__':
    main()