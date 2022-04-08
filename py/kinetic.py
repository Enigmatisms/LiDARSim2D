import numpy as np
import matplotlib.pyplot as plt

def loadFromFile(path:str) -> np.array:
    with open(path, 'r') as file:
        raw_lines = file.readlines()
        raw_lines = [line[:-1] for line in raw_lines]
        split_lines = [line.split(' ') for line in raw_lines]
        data = [[float(val) for val in line] for line in split_lines]
    return np.array(data)

# 前两列：平移（delta值），第三列（角度变化值)， 第四列：实际速度，第五列：运动控制信息
def plotKinetic(data:np.array):
    xs = np.arange(data.shape[0])
    name = ('translation x', 'translation y', 'delta angle')
    plt.figure(0)
    for i in range(3):
        plt.subplot(3, 1, i + 1)
        plt.plot(xs, data[:, i] * (1.0 if i < 2 else 5.0), label = name[i])
        if i < 2:
            plt.plot(xs, data[:, -2], label = 'actual speed')
            plt.plot(xs, data[:, -1], label = 'Control')
        plt.grid(axis = 'both')
        plt.legend()
    plt.figure(1)
    for i in range(2):
        diff_tf = data[1:, 2 * i + 3] - data[:-1, 2 * i + 3]
        diff_cv = data[1:, 2 * i + 4] - data[:-1, 2 * i + 4]
        xs2 = np.arange(diff_tf.shape[0])
        plt.subplot(3, 1, i + 1)
        plt.plot(xs2, diff_tf, label = name[i] + " tf")
        plt.plot(xs2, diff_cv, label = name[i])
        plt.plot(xs2, data[:-1, -2], label = 'actual speed')
        plt.plot(xs2, data[:-1, -1], label = 'Control')
        plt.grid(axis = 'both')
        plt.legend()
    plt.subplot(3, 1, 3)
    # plt.plot(xs, data[:, -3], label = 'Abs angle')
    angle_diff = data[1:, -3] - data[:-1, -3]
    angle_diff[angle_diff > np.pi] -= 2 * np.pi
    angle_diff[angle_diff < -np.pi] += 2 * np.pi
    plt.plot(xs2, angle_diff, label = 'Angle diff')
    plt.grid(axis = 'both')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    data = loadFromFile("../bags/kinetic.log")
    plotKinetic(data)
