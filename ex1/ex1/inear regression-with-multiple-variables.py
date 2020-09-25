import numpy as np
import matplotlib.pyplot as plt


J_value = []


def normalization(X):
    shape = X.shape
    means = np.zeros((shape[1], 1))
    stds = np.zeros((shape[1], 1))
    for col in range(shape[1]):
        means[col] = np.mean(X[:, col])
        stds[col] = np.std(X[:, col])
        X[:, col] = (X[:, col] - means[col]) / stds[col]
    
    return X, means.T, stds.T


def cost_function(X, y, theta, m):
    a = np.dot(X, theta) # h(X)
    b = a - y[:][np.newaxis].T
    return np.dot(b.T, b)[0][0] / 2 / m


def gradient_descent(X, y, iteration=1500, alpha=0.01):
    shape = X.shape
    m = shape[0] # num of samples
    theta = np.zeros((1, shape[1]+1)).T # initiate theta

    X = np.concatenate((np.ones((m, 1)), X), axis=1) # add a colum of one to X

    J_value.append(cost_function(X, y, theta, m))
    print(shape[1] + 1)
    for i in range(iteration):
        theta_temp = theta
        a = np.dot(X, theta_temp) # h(X)
        b = a - y[:][np.newaxis].T

        for col in range(shape[1] + 1):
            theta[col] = theta_temp[col] - alpha * np.dot(b.T, X[:, col]) / m

        J_value.append(cost_function(X, y, theta, m))

    return theta.T[0]


def normal_equations(X,y):
    shape = X.shape
    m = shape[0] # num of samples

    X = np.concatenate((np.ones((m, 1)), X), axis=1) # add a colum of one to X

    a = np.dot(X.T, X)
    b = np.linalg.pinv(a)
    c = np.dot(b, X.T)

    return np.dot(c, y)
        


def main():
    data = np.loadtxt('./ex1/ex1/ex1data2.txt', delimiter=',')
    X = data[:, :-1]
    y = data[:, -1]
    
    theta_1 = normal_equations(X,y) # solve by normal equation

    X, means, stds = normalization(X) # normalize before gradient descent
    theta_2 = gradient_descent(X, y, iteration=500)

    print('Theta solved by normal equation is: ', theta_1)
    print('Theta solved by gradient descent is: ', theta_2)

    x = np.array([1., 1650, 3])
    print('Price of the 1650-square-foot house with 3 bedrooms is (normal equation): ', np.dot(x, theta_1.T))
    x[1:] = (x[1:] - means) / stds
    print('Price of the 1650-square-foot house with 3 bedrooms is (gradient descent): ', np.dot(x, theta_2.T))

    # plot J
    plt.plot(J_value)
    plt.xlabel('Number of iterations')
    plt.ylabel('Cost J')
    plt.show()



if __name__ == '__main__':
    main()
    

