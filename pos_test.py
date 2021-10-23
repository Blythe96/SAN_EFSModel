"""
台风风眼轨迹观察
"""
import os
import torch
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def draw(data):
    data = np.array(data)
    x1 = data[:, 0]
    y1 = data[:, 1]
    plt.plot(x1, y1, color='red', marker="^", label='label')
    plt.show()


def read_all(dir_path):
    result = []
    for dirname, _, filenames in os.walk(dir_path):
        for filename in filenames:
            if filename.endswith(".npy"):
                g, i, lat, lon, *_ = filename.split("_")
                result.append([int(g), int(i), float(lat), float(lon)])
    result = pd.DataFrame(result, columns=["g", "i", "lat", "lon"])
    all = []
    for g in result.groupby("g").size().keys():
        data = result[result['g'] == g].sort_values('i')
        y = torch.tensor(data[['lat', 'lon']].to_numpy(), dtype=torch.float32)
        draw(y)
        print(y.tolist())
    return all


if __name__ == "__main__":
    read_all('/home/zmp/projects/data_acquisition/tasks/imgs')
