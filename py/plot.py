import numpy as np
import matplotlib.pyplot as plt

def plotData(arr:np.array):
    length = arr.shape[0]
    plt.subplot(3, 1, 1)
    arr = arr.reshape([-1, 4])
    xs = np.arange(length)
    plt.scatter(xs, arr[:, 0], c = 'k', s = 7)
    plt.plot(xs, arr[:, 0], c = 'b')
    plt.grid(axis = 'both')
    plt.subplot(3, 1, 2)
    for i in range(1, length):
        arr[i, 0] += arr[i - 1, 0]
    plt.scatter(xs, arr[:, 0], c = 'k', s = 7)
    plt.plot(xs, arr[:, 0], c = 'r')
    plt.grid(axis = 'both')
    plt.subplot(3, 1, 3)
    plt.scatter(xs, arr[:, 2], c = 'k', s = 7)
    plt.plot(xs, arr[:, 2], c = 'r')
    plt.plot(xs, arr[:, 3], c = 'b')
    plt.grid(axis = 'both')

if __name__ == "__main__":
    origin = np.loadtxt("../data/data.txt", dtype = float, delimiter = ",")
    plotData(origin)
    plt.show()