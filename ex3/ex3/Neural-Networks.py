import numpy as np
from scipy import io
from matplotlib import pyplot as plt
from scipy.optimize import optimize


def main():
    data = io.loadmat('./ex3/ex3/ex3data1.mat')
    X = data['X']
    y = data['y']
    np.info(y)


if __name__ == '__main__':
    main()
