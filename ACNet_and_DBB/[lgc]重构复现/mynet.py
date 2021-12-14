import torch
import torch.nn as nn
from diversebranchblock import DiverseBranchBlock

class OriCNNnet(nn.Module):
    """
    一个原始的CNN
    """
    def __init__(self):
        super(OriCNNnet, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.mlp1 = nn.Linear(2 * 2 * 64, 100)
        self.mlp2 = nn.Linear(100, 10)
    
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.mlp1(x.view(x.size(0),-1))
        x = self.mlp2(x)
        return x



class DbbCNNnet(nn.Module):
    """
    使用DBB:
        在原始CNN(OriCNNnet)的基础上, 把conv-bn组合替换为DiverseBranchBlock(..., deploy=False)
        参数设置和OriCNNnet保持一致即可
    """
    def __init__(self):
        super(DbbCNNnet, self).__init__()
        self.conv1 = nn.Sequential(
            DiverseBranchBlock(in_channels=3, out_channels=16, kernel_size=3, stride=2, padding=1),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            DiverseBranchBlock(in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=1),
            nn.ReLU()
        )
        self.conv3 = nn.Sequential(
            DiverseBranchBlock(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1),
            nn.ReLU()
        )
        self.conv4 = nn.Sequential(
            DiverseBranchBlock(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1),
            nn.ReLU()
        )
        self.mlp1 = nn.Linear(2 * 2 * 64, 100)
        self.mlp2 = nn.Linear(100, 10)
    
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.mlp1(x.view(x.size(0),-1))
        x = self.mlp2(x)
        return x



if __name__ == "__main__":
    inp = torch.rand((32, 3, 32, 32))
    mynet1 = OriCNNnet()
    mynet2 = DbbCNNnet()
    outp1 = mynet1(inp)
    outp2 = mynet2(inp)
    print()