from env import *


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(  # input shape (1,64,64)
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            # shape (32,32,32)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
            # shape (64,16,16)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            # shape (32,21,26)
            nn.ReLU(),
            nn.MaxPool2d(2),
            # shape （128，8，8）
        )

        self.out = nn.Sequential(
            nn.Linear(128 * 64, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64,8),
            nn.ReLU(),
            nn.Linear(8,2),
        )

    def forward(self, h):
        output = []
        b = h.shape[0]
        for i in range(out_step):
            x = h[:, i, :, :]
            x = self.conv1(x.view(b, 1, width, height))
            x = self.conv2(x)
            x = self.conv3(x)
            x = x.view(h.size(0), -1)
            x = self.out(x)
            output.append(x.view(b, 1, 2))
        out = torch.cat(output, dim=1)
        return out

class D2NL(nn.Module):
    def __init__(self):
        super(D2NL, self).__init__()
        self.l = nn.Sequential(
            nn.Linear(width * height, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 4),
            nn.ReLU(),
            nn.Linear(4, 2),
        )

    def forward(self, x):
        # B,S,1,64,64
        b, s, c, w, h = x.shape
        return self.l(x.view(b, s, c * w * h))
