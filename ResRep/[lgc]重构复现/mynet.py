import torch
import torch.nn as nn
from compactor import CompactorLayer


class ResRepBlock(nn.Module):
    def __init__(self, in_features, out_features):
        super(ResRepBlock, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.deploy = False
        self.rrb_deploy = None
        self.conv_3_3 = nn.Conv2d(
            in_channels=in_features, out_channels=out_features,
            kernel_size=3, stride=2, padding=1
        )
        self.bn = nn.BatchNorm2d(out_features)
        self.compactor = CompactorLayer(num_features=out_features, conv_idx=0)

    def forward(self, x):
        if self.deploy:
            return self.rrb_deploy(x)
        x = self.conv_3_3(x)
        x = self.bn(x)
        y = self.compactor(x)
        return y

    def get_equivalent_kernel_bias(self):
        weight_3_3 = self.conv_3_3.weight
        bias_3_3 = self.conv_3_3.bias
        running_mean = self.bn.running_mean
        running_var = self.bn.running_var
        gamma = self.bn.weight
        beta = self.bn.bias
        eps = self.bn.eps

        # fuse 3×3Conv and BN
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        weight_3_3_bn, bias_3_3_bn = weight_3_3 * t, bias_3_3 + beta - running_mean * gamma / std

        # 3×3Conv transpose
        weight_3_3_bn_T = weight_3_3_bn.transpose(0, 1)
        bias_3_3_bn_T = bias_3_3_bn.transpose(0, 1)

        # 再卷积
        out = self.compactor(weight_3_3_bn_T)
        out_T = out.transpose(0, 1)
        return out_T, bias_3_3_bn_T

    def switch_to_deploy(self):
        """
        使用一个等价代换的 rbr_reparam (卷积层)代替 rbr_dense, rbr_1x1, rbr_identity, id_tensor
        设置 self.deploy = True
        """
        if hasattr(self, 'rrb_deploy'):
            return
        kernel, bias = self.get_equivalent_kernel_bias()
        self.rrb_deploy = nn.Conv2d(
            in_channels=self.in_features,
            out_channels=self.out_features,
            kernel_size=3, stride=2, padding=1, bias=True
        )
        self.rrb_deploy.weight.data = kernel
        self.rrb_deploy.bias.data = bias
        for para in self.parameters():
            para.detach_()
        self.__delattr__('conv_3_3')
        self.__delattr__('bn')
        self.__delattr__('conv_1_1')
        self.deploy = True




class ResRepNet(nn.Module):
    def __init__(self):
        super(ResRepNet, self).__init__()
        self.conv1 = nn.Sequential(
            ResRepBlock(in_features=3, out_features=16),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            ResRepBlock(in_features=16, out_features=32),
            nn.ReLU()
        )
        self.conv3 = nn.Sequential(
            ResRepBlock(in_features=32, out_features=32),
            nn.ReLU()
        )
        self.conv4 = nn.Sequential(
            ResRepBlock(in_features=32, out_features=64),
            nn.ReLU()
        )
        self.conv5 = nn.Sequential(
            ResRepBlock(in_features=64, out_features=64),
            nn.ReLU()
        )
        self.mlp1 = nn.Linear(1 * 1 * 64, 100)
        self.mlp2 = nn.Linear(100, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.mlp1(x.view(x.size(0), -1))
        x = self.mlp2(x)
        return x
