import torch
from torch import nn
import torch.nn.functional as F


class Cifar100Net(nn.Module):
    def __init__(self):
        super(Cifar100Net, self).__init__()
        self.net = nn.Sequential(
            # conv output size: [(width - kernel + 2padding)/stride]+1  32-3+2+1=32
            nn.Conv2d(3, 64, 3, padding=1),  # 输入通道数为3，输出通道数为64， filter=3*3
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.MaxPool2d(2, 2),  # (32-2+0)/2+1=16
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        self.fc1 = nn.Linear(16 * 16 * 128, 1024)  # flatten length * width * channels
        self.drop1 = nn.Dropout2d()
        self.fc2 = nn.Linear(1024, 128)

    def forward(self, x):
        net_x = self.net(x)
        x = net_x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.drop1(x)
        x = self.fc2(x)

        return x


if __name__ == '__main__':
    cifar100net = Cifar100Net().cuda()
    data = torch.randn(10, 3, 32, 32).cuda()
    # print(cifar100net)
    cifar100net(data)
