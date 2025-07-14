import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable

# TODO: 研究变量替换函数以正确强制动作边界
class DiagonalGaussian(nn.Module):
    """
    对角高斯分布类，用于建模连续动作空间的随机策略
    （注：类名虽为DiagonalGaussian，但实际实现为各维度独立的高斯分布，即对角协方差矩阵）
    """
    def __init__(self, num_outputs, init_std=1, learn_std=True):
        """
        参数:
            num_outputs: 动作空间维度（输出维度）
            init_std: 初始标准差（用于初始化logstd参数）
            learn_std: 是否将标准差作为可学习参数（True则通过训练调整）
        """
        super(DiagonalGaussian, self).__init__()

        # 初始化log标准差参数（通过指数转换为实际标准差）
        # 形状为(1, num_outputs)，与动作维度匹配
        self.logstd = nn.Parameter(
            torch.ones(1, num_outputs) * np.log(init_std),  # 初始值为log(init_std)
            requires_grad=learn_std  # 是否可学习
        )

        self.learn_std = learn_std  # 记录是否学习标准差

    def forward(self, x):
        """
        前向传播，计算高斯分布的均值和标准差
        参数:
            x: 输入张量（通常为策略网络输出的均值，形状: [batch_size, num_outputs]）
        返回:
            mean: 高斯分布的均值（与输入x相同，形状: [batch_size, num_outputs]）
            std: 高斯分布的标准差（由logstd指数化得到，形状: [1, num_outputs]）
        """
        mean = x  # 均值直接使用输入x（假设x是策略网络输出的均值）

        std = self.logstd.exp()  # 标准差 = e^(logstd)
        
        return mean, std

    def sample(self, x, deterministic):
        """
        从高斯分布中采样动作（或返回确定性动作）
        参数:
            x: 输入均值（形状: [batch_size, num_outputs]）
            deterministic: 是否返回确定性动作（True则返回均值，False则采样）
        返回:
            action: 生成的动作（形状: [batch_size, num_outputs]）
        """
        if deterministic is False:
            # 非确定性模式：从高斯分布中采样动作
            action = self.evaluate(x).sample()
        else:
            # 确定性模式：直接返回均值作为动作
            action, _ = self(x)

        return action

    def evaluate(self, x):
        """
        构建并返回高斯分布对象（用于计算概率、对数概率等）
        参数:
            x: 输入均值（形状: [batch_size, num_outputs]）
        返回:
            torch.distributions.Normal: 高斯分布对象
        """
        mean, std = self(x)  # 获取均值和标准差
        return torch.distributions.Normal(mean, std)  # 构建正态分布