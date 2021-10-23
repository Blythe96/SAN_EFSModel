from torch.utils.data import Dataset, ConcatDataset, DataLoader
from env import *
import numpy as np
import os
import pandas as pd
import torch
from label_evaluate import normalization, lon_lat_to_graph2, lon_lat_to_graph, move_window
import pickle

from PIL import Image
import torchvision.transforms as transforms


def resize(data, size=64):
    mode = Image.fromarray(data, mode="F")
    transform1 = transforms.Compose([
        transforms.Scale(size),
        transforms.CenterCrop((size, size)),
        transforms.ToTensor()
    ])
    mode = transform1(mode)
    return mode


class SupervisionDataSet(Dataset):

    def __init__(self, x):
        total, c, w, h = x.shape
        self.x = x
        # self.y = x[:, 1, :, :].view(total, 1, w, h)
        self.y = x
        self.__len = max(total - in_step - out_step, 0)

    def __len__(self):
        return self.__len

    def __getitem__(self, index):
        x = self.x.clone()
        y = self.y.clone()
        y = y[index + in_step:index + in_step + out_step]
        x = x[index:index + in_step]

        # cnn
        # x = x[index]
        # y = y[index]
        return x, y


def read_all(dir_path):
    result = []
    for dirname, _, filenames in os.walk(dir_path):
        for filename in filenames:
            if filename.endswith(".npy"):
                full_path = os.path.join(dirname, filename)
                g, i, lat, lon, *_ = filename.split("_")
                d = np.load(full_path)
                # d.resize((101, 101))
                d = d.reshape((1, d.shape[0], d.shape[1]))
                # d = resize(d, width)
                # d = d.view()
                result.append([int(g), int(i), float(lat), float(lon), d])
    result = pd.DataFrame(result, columns=["g", "i", "lat", "lon", "p"])
    all = []
    for g in result.groupby("g").size().keys():
        data = result[result['g'] == g].sort_values('i')

        y = torch.tensor(data[['lat', 'lon']].to_numpy(), dtype=torch.float32)
        x = torch.tensor(np.vstack(data['p'].values), dtype=torch.float32)
        total, w, h = x.shape

        ch0 = torch.cat([resize(d.numpy(), 256) for d in x])
        ch1 = lon_lat_to_graph(y.clone(), 256, 256)
        x = torch.cat([
            ch0.view(total, 1, 256, 256).mul(ch1.view(total, 1, 256, 256)),
        ], dim=1)
        all.append(SupervisionDataSet(x))
        # for d in x:
        #     all.append(SupervisionDataSet(d))
    all = ConcatDataset(all)
    return all


def gen_grid_data_set(dir_path):
    all = read_data_from_cache()
    if all is not None:
        all_length = len(all)
        train_size = int(all_length * 0.9)
        test_size = all_length - train_size
        train_data, test_data = torch.utils.data.random_split(all, [train_size, test_size])
        return train_data, test_data
    all = read_all(dir_path)
    save_data(all)
    all_length = len(all)
    train_size = int(all_length * 0.95)
    test_size = all_length - train_size
    train_data, test_data = torch.utils.data.random_split(all, [train_size, test_size])
    return train_data, test_data


def save_data(data):
    with open("data.bin", "wb") as data_file:
        pickle.dump(data, data_file)
    print("save data finish")


def read_data_from_cache():
    if not os.path.exists('data.bin'):
        return None
    with open("data.bin", "rb") as data_file:
        print(data_file)
        data = pickle.load(data_file)
    print("read data from cache")
    return data


def get_loader():
    train_data_set, test_data_set = gen_grid_data_set('/home/zmp/projects/data_acquisition/tasks/imgs')
    train_data_loader = DataLoader(
        dataset=train_data_set,
        batch_size=batch_size,
        shuffle=False,
    )
    test_data_loader = DataLoader(
        dataset=test_data_set,
        batch_size=batch_size,
        shuffle=False
    )
    return train_data_loader, test_data_loader

# if __name__ == "__main__":
#     gen_grid_data_set('/home/zmp/projects/data_acquisition/tasks/imgs')
#     print('finish')
