from loader import get_loader
from torch import nn
from env import *


class View(nn.Module):

    def __init__(self, v):
        super(View, self).__init__()
        self.v = v

    def forward(self, x):
        return x.view(-1, self.v)


class Model(nn.Module):
    def __init__(self, w, h):
        super(Model, self).__init__()
        self.l0 = nn.Sequential(
            # 310,381
            nn.Conv2d(in_channels=1, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            View(128 * 38 * 47),
            nn.Linear(128 * 38 * 47, 256),
            nn.Sigmoid(),
            nn.Linear(256, 128),
            nn.Sigmoid(),
            nn.Linear(128, 64),
            nn.Sigmoid(),
            nn.Linear(64, 32),
            nn.Sigmoid(),
            nn.Linear(32, 16),
            nn.Sigmoid(),
            nn.Linear(16, 8),
            nn.Sigmoid(),
            nn.Linear(8, 4),
            nn.Sigmoid(),
            nn.Linear(4, 2),
        )

    def forward(self, x):
        return self.l0(x)


def d_loss(p, y):
    """
    p: shape *,2
    y: shape *,2
    """
    x1 = p[:, 0]
    x2 = y[:, 0]
    y1 = p[:, 1]
    y2 = y[:, 1]
    return torch.mean((x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2))


if __name__ == "__main__":
    tran_data_loader, _ = get_loader()
    model = Model(310, 381)
    optim = torch.optim.Adam(model.parameters(), lr=lr)  # 优化器
    model.to(device)
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.xavier_uniform_(m.weight)
    for e in range(20):
        for id, data in enumerate(tran_data_loader):
            x, y = data
            b, w, h = x.shape
            x = x.view(b, 1, w, h)
            x = x.to(device)
            y = y.to(device)
            p = model(x)
            loss = d_loss(p, y)
            print("e: %d/20 [%d/%d] loss:" % (e, id, len(tran_data_loader)), loss.item())
            optim.zero_grad()
            loss.backward()
            # torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=10.0)
            optim.step()
