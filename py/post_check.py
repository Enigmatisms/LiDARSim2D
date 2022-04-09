import numpy as np
import matplotlib.pyplot as plt

def loadFromFile(path:str) -> np.array:
    with open(path, 'r') as file:
        raw_lines = file.readlines()
        raw_lines = [line[:-1] for line in raw_lines]
        split_lines = [line.split(' ') for line in raw_lines]
        data = [[float(val) for val in line] for line in split_lines]
    return np.array(data)

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

# column: (px, py, angle, vx, vy, va)
def plotOdom(data:np.array, gt:np.array):
    dx = np.cos(data[:, 2]) / 2.
    dy = np.sin(data[:, 2]) / 2.
    gt_dx = np.cos(gt[:, -3]) / 2.
    gt_dy = np.sin(gt[:, -3]) / 2.
    plt.figure(0)
    plt.plot(data[:, 0], data[:, 1], label = 'odom', color = '#FF4500')
    for i in range(data.shape[0]):
        plt.arrow(data[i, 0], data[i, 1], dx[i], dy[i], color = '#FF4500', head_width = 0.05)
    start_x, start_y = gt[0, 3] / 50., gt[0, 5] / 50.
    plt.plot(gt[:, 3] / 50. - start_x, gt[:, 5] / 50. - start_y, label = 'gt', color = '#29A0B1')
    for i in range(gt.shape[0]):
        plt.arrow(gt[i, 3] / 50. - start_x, gt[i, 5] / 50. - start_y, gt_dx[i], gt_dy[i], color = '#29A0B1', head_width = 0.05)
    plt.legend()
    plt.grid(axis = 'both')
    plt.figure(1)
    for i in range(3):
        plt.subplot(3, 1, 1 + i)
        xs1, xs2 = np.arange(gt.shape[0]), np.arange(data.shape[0])
        plt.plot(xs1, gt[:, i] * (1 if i < 2 else 100), label = 'gt')
        plt.plot(xs2, data[:, i + 3], label = 'odom')
        plt.legend()
        plt.grid(axis = 'both')
    plt.figure(2)
    plt.plot(xs2, data[:, -2], label = 'IMU duration')
    plt.plot(xs2, data[:, -1] / 1000, label = 'Tictoc duration')
    plt.legend()
    plt.grid(axis = 'both')
    plt.show()

def plotOdomData():
    odom = loadFromFile("../bags/odom.log")
    gt = loadFromFile("../bags/kinetic.log")
    plotOdom(odom, gt)


def plotKinectData():
    data = loadFromFile("../bags/kinetic.log")
    plotKinetic(data)

if __name__ == "__main__":
    plotOdomData()
