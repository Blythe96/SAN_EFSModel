from loader import *
from label_evaluate import save_image
import torch

if __name__ == "__main__":
    all = read_all('/home/zmp/projects/data_acquisition/tasks/imgs')
    l = 15
    i = 0
    result = []
    for data in all:
        if i > l:
            break
        x, y = data
        for d in range(16):
            save_image(x[d], torch.zeros(1), "x", 1)
        i += 1
    # t, w, h = data.shape
    # save_image(data.view(t, 1, w, h), torch.zeros(1), "x", 8)

    # data = torch.ones(100, 100)
    # # data = torch.tensor(hit(data, 0, 0))
    # data = torch.tensor(hit(data, 20, 20))
    # save_image(data.view(1, 1, 100, 100), torch.zeros(1), "x", 8)
    # print(data)
