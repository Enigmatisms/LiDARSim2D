import numpy as np
import matplotlib.pyplot as plt

def covolution(arr:np.array):
    length = len(arr)
    res = np.zeros_like(arr)
    for i in range(length):
        if i > 0 and i < length - 1:
            res[i] = abs(arr[i + 1] - arr[i - 1])
        elif i == 0:
            res[0] = abs(arr[1] - arr[0])
        else:
            res[i] = abs(arr[-1] - arr[-2])
    return res

if __name__ == "__main__":
    file_name = "../maps/range.txt"
    file = open(file_name, "r")
    lines = file.readlines()
    length = len(lines)
    angle_incre = 0.0087266
    while True:
        range_num = input("Range to visualize (0~%d): "%(length - 1))
        if range_num.isdigit() == False:
            break
        range_num = int(range_num)
        line = lines[range_num]
        raw_nums = line.split(",")
        raw_nums.pop()
        px, py, start_id = raw_nums[:3]
        px, py, start_id = (float(px), float(py), int(start_id))
        raw_nums = raw_nums[3:]
        ys = np.array([float(x) for x in raw_nums])
        xs = np.arange(len(ys))
        pts = [[r * np.cos((i + start_id) * angle_incre - np.pi) + px, r * np.sin((i + start_id) * angle_incre - np.pi) + py] for i, r in enumerate(ys)]
        pts = np.array(pts)
        ys2 = covolution(ys)
        plt.figure(0)
        plt.plot(xs, ys)
        plt.plot(np.arange(len(ys2)), ys2)
        plt.figure(1)
        plt.scatter(pts[:, 0], pts[:, 1], c = 'b', s = 8)
        plt.show()
    file.close()