import torch
import torch.nn as nn
from diversebranchblock import DiverseBranchBlock


if __name__ == '__main__':
    x = torch.randn(1, 32, 56, 56)
    for k in (3, 5):
        for s in (1, 2):
            # 分别建立 (kernel=3, stride=1); (kernel=3, stride=2); 
            # (kernel=5, stride=1); (kernel=5, stride=2) 四种DBB模型
            dbb = DiverseBranchBlock(
                in_channels=32, out_channels=64, 
                kernel_size=k, stride=s, padding=k//2,
                groups=2, deploy=False
            )
            
            # 初始化相关参数
            for module in dbb.modules():
                if isinstance(module, torch.nn.BatchNorm2d):
                    nn.init.uniform_(module.running_mean, 0, 0.1)
                    nn.init.uniform_(module.running_var, 0, 0.1)
                    nn.init.uniform_(module.weight, 0, 0.1)
                    nn.init.uniform_(module.bias, 0, 0.1)
            
            dbb.eval()  # 测试状态
            print(dbb)  # 打印训练时dbb模型构成
            train_y = dbb(x)  # dbb推理结果
            dbb.switch_to_deploy()  # dbb经过等价转换成为单串行模型
            deploy_y = dbb(x)  # 等价转换后的dbb进行推理的结果
            print(dbb)  # 打印推理时dbb模型构成
            print('========================== The diff is', ((train_y - deploy_y) ** 2).sum())  # 打印总方差