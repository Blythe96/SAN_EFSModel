from label_evaluate import evaluate, un_normalization, distance, toReal01
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from loader import get_loader


def avg(arr):
    return torch.mean(torch.tensor(arr)).item()


lrg = {}


def record(k, v):
    if k not in lrg:
        lrg[k] = []
    lrg[k].append(v)


def avg_record():
    result = {}
    for k in lrg.keys():
        result[k] = avg(lrg[k])
    return result


def show(r):
    for k in r.keys():
        print(k, r[k])


def get_avg_pos(data):
    result = []
    for g in data:
        x, y = 0, 0
        l = 0
        for i in range(len(g)):
            r = g[i]
            for j in range(len(r)):
                if r[j] > 0:
                    l += 1
                    x = i
                    y = j
        result.append([x / l / 256, y / l / 256])
    return result


def get_max_pos(data):
    result = []
    for g in data:
        x, y = 0, 0
        l = 0
        mx = torch.max(g)
        for i in range(len(g)):
            r = g[i]
            for j in range(len(r)):
                if r[j] == mx:
                    l += 1
                    x += i
                    y += j
        result.append([x / l / 256, y / l / 256])
    return result


def draw():
    # 创建图并命名
    plt.figure('Line fig')
    ax = plt.gca()
    # 设置x轴、y轴名称
    ax.set_xlabel('x')
    ax.set_ylabel('y')

    # 画连线图，以x_list中的值为横坐标，以y_list中的值为纵坐标
    # 参数c指定连线的颜色，linewidth指定连线宽度，alpha指定连线的透明度
    ax.plot(l[:, 0], l[:, 1], color='r', linewidth=1, alpha=0.6)

    plt.show()


if __name__ == "__main__":
    model_path = "/home/zmp/projects/undnow/save_model/feat-task-prediction-intensity/full_48_0.000226.pth.tar"
    net = torch.load(model_path)
    _, validLoader = get_loader()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # 设备选取
    with torch.no_grad():
        net.eval()
        t = tqdm(validLoader, leave=False, total=len(validLoader))
        for i, (inputVar, targetVar) in enumerate(t):
            inputs = inputVar.to(device)
            label = targetVar.to(device)
            pred = net(inputs)
            p = torch.tensor(get_max_pos(pred.clone().reshape(-1, 256, 256)), dtype=torch.float)
            l = torch.tensor(get_max_pos(label.clone().reshape(-1, 256, 256)), dtype=torch.float)

            # mse, rmse, mae, min_diff, max_diff = evaluate(pred.clone().reshape(-1, 64, 64),
            #                                               label.clone().reshape(-1, 64, 64))
            mse, rmse, mae, min_diff, max_diff = evaluate(torch.tensor(l, dtype=torch.float),
                                                          torch.tensor(p, dtype=torch.float))
            t0 = un_normalization(l)
            t1 = un_normalization(p)
            u_mse, u_rmse, u_mae, u_min_diff, u_max_diff = evaluate(t0, t1)
            e1, e2 = distance(toReal01(p), toReal01(l))
            # if avg(e2) < 200:
            print()
            record("mse", mse)
            record("rmse", rmse)
            record("mae", mae)
            record("min_diff", min_diff)
            record("max_diff", max_diff)
            record("u_mse", u_mse)
            record("u_rmse", u_rmse)
            record("u_mae", u_mae)
            record("u_min_diff", u_min_diff)
            record("u_max_diff", u_max_diff)
            record("distance", avg(e2))
            print("label:", t0.tolist())
            print("pred :", t1.tolist())
            print("mse: %f,rmse: %f,mae: %f,min_diff: %f,max_diff: %f" % (
                mse, rmse, mae, min_diff, max_diff))
            print("u_mse: %f,u_rmse: %f,u_mae: %f,u_min_diff: %f,u_max_diff: %f,dis: %f" % (
            u_mse, u_rmse, u_mae, u_min_diff, u_max_diff, avg(e2)))
        show(avg_record())

"""
pred lat: 22.442779541015625
pred lon: 25.169029235839844
real lat: 21.0
real lon: 16.80001449584961
E1 1689.1090087890625
E2 nan
"""
