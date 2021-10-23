from env import *
import torch
import torchvision
import os
import time
from scipy.ndimage import gaussian_filter

import math

min_lat_diff = 0.
max_lat_diff = 65.
min_lon_diff = 100.
max_lon_diff = 180.

# min_lat_diff = 100
# max_lat_diff = 180
# min_lon_diff = 0
# max_lon_diff = 65

from PIL import Image, ImageDraw


def hit(data, x, y):
    s = 4
    dt = data.numpy()
    d = ImageDraw.Draw(Image.fromarray(dt), mode="F")
    x = max([x - s // 2, 0])
    y = max([y - s // 2, 0])
    d.rectangle((x, y, x + s, y + s), 0)
    return d.im


def move_window(data, ww, wh, w_step=1, h_step=1):
    """
    data: shape n,w,h
    """
    *_, width, height = data.shape
    result = []
    for i in range((width - ww) // w_step):
        for j in range((height - wh) // h_step):
            result.append(data[:, i * w_step:i * w_step + ww,
                          j * h_step:j * h_step + wh].view(*_, 1, ww, wh))
    return torch.cat(result, dim=1).permute(1, 0, 2, 3)


if __name__ == "__main__":
    d = torch.zeros(20, 300, 300)
    t = move_window(d, 64, 64)
    print(t.shape)


def lon_lat_to_graph(y, w, h, sigma=8):
    """
    经纬度转密度图
    :param y: shape n,2
    :return: n,w,h
    """
    result = []
    for data in y:
        lat, lon = data[0], data[1]
        g = torch.zeros((w, h), dtype=torch.float)
        x, y = (lat - min_lat_diff) / (max_lat_diff - min_lat_diff) * w, (lon - min_lon_diff) / (
                max_lon_diff - min_lon_diff) * h
        g[int(x)][int(y)] = torch.tensor(1., dtype=torch.float)
        g = gaussian_filter(g, sigma=sigma)
        result.append(torch.tensor(g).view(1, w, h))
    return torch.cat(result, dim=0)


def lon_lat_to_graph2(x, y):
    """
    经纬度转标签图
    :param y: shape n,2
    :return: n,w,h
    """
    result = []
    i = 0
    for g in x:
        data = y[i]
        lat, lon = data[0], data[1]
        _x, _y = (lat - min_lat_diff) / (max_lat_diff - min_lat_diff) * width, (lon - min_lon_diff) / (
                max_lon_diff - min_lon_diff) * height
        g = hit(g, _x, _y)
        result.append(torch.tensor(g).view(1, width, height))
        i += 1
    return torch.cat(result, dim=0)


def find(g):
    for i in range(width):
        for j in range(height):
            if g[i][j] == 1.:
                return i, j


def graph_to_lon_lat(graphs):
    """
    标签图转经纬度
    :param graphs: shape n,width,height
    :return:
    """
    result = []
    for g in graphs:
        x, y = find(g)
        lat = x * (max_lat_diff - min_lat_diff) / width + min_lat_diff
        lon = y * (max_lon_diff - min_lon_diff) / height + min_lon_diff
        result.append([lat, lon])
    return torch.tensor(result)


def normalization(data):
    """
    data.shape (n,2)
    """
    d = data.clone()
    d[:, 0] = (d[:, 0] - min_lat_diff) / (max_lat_diff - min_lat_diff)
    d[:, 1] = (d[:, 1] - min_lon_diff) / (max_lon_diff - min_lon_diff)
    return d


def un_normalization(data):
    """
    data.shape (n,2)
    """
    d = data.clone()
    d[:, 0] = d[:, 0] * (max_lat_diff - min_lat_diff) + min_lat_diff
    # d[:, 0] = d[:, 0] * 10
    d[:, 1] = d[:, 1] * (max_lon_diff - min_lon_diff) + min_lon_diff
    return d


def toReal01(data):
    d = data.clone()
    d = un_normalization(d)
    d = d * math.pi / 180
    return d


def distance(pred, real):
    '''
    :param pred:  shape (n,2)
    :param real:  shape (n,2)
    :return: (n,1)
    '''
    R = 6357
    latPred = pred[:, 0]
    lonPred = pred[:, 1]
    latReal = real[:, 0]
    lonReal = real[:, 1]
    E1 = 2 * R * torch.asin(
        torch.sqrt(
            torch.sin(
                torch.pow((latPred - latReal) / 2, 2)
            )
            + torch.cos(latReal) * torch.cos(latPred) *
            torch.sin(torch.pow((lonPred - lonReal) / 2, 2))
        )
    )
    E2 = 2 * R * torch.asin(
        torch.sqrt(
            torch.pow(torch.sin((latPred - latReal) / 2), 2) + torch.cos(latReal) * torch.cos(latPred) * torch.pow(
                (lonPred - lonReal) / 2, 2)
        )
    )
    # print("E1", E1.tolist())
    # print("E2", E2.tolist())

    return E1, E2


# def evaluate(pred, real):
#     '''
#     :param pred: shape(n,2)
#     :param real: shape(n,2)
#     :return: mae,mse,rmse
#     '''
#     mse = torch.mean((pred - real) ** 2)
#     rmse = torch.sqrt(mse)
#     mae = torch.mean(torch.abs(pred - real))
#     min_diff = (torch.abs(pred - real).min())
#     max_diff = (torch.abs(pred - real).max())
#     return mse.item(), rmse.item(), mae.item(), min_diff, max_diff

def evaluate(pred, real):
    '''
    :param pred: shape(n,2)
    :param real: shape(n,2)
    :return: mae,mse,rmse
    '''
    mse = torch.mean((pred - real) ** 2)
    rmse = torch.sqrt(mse)
    mae = torch.mean(torch.abs(pred - real))
    min_diff = (torch.abs(pred - real).min())
    max_diff = (torch.abs(pred - real).max())
    return mse, rmse, mae, min_diff, max_diff


def save_image(imgs, loss, pre, nrow=8):
    save_path = os.path.join("procs", "%s_%f_%s.png" % (time.time(), loss.item(), pre))
    img = torchvision.utils.make_grid(imgs, normalize=True, nrow=nrow, )
    torchvision.utils.save_image(img, save_path, )
