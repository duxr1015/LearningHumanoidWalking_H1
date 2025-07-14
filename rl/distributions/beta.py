import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

# TODO: 扩展这些类以支持任意动作边界
"""一个Beta分布，但其概率密度函数（pdf）被缩放至(-1, 1)范围"""
class BoundedBeta(torch.distributions.Beta):
    def log_prob(self, x):
        """
        计算动作x的对数概率（重写父类方法以适配(-1,1)范围的动作）
        参数:
            x: 输入动作（范围(-1,1)）
        返回:
            动作x的对数概率
        说明:
            Beta分布的原生定义域是(0,1)，因此需要将(-1,1)的动作x转换为(0,1)：
            (x + 1)/2 可将(-1,1)映射到(0,1)，再调用父类的log_prob计算概率
        """
        return super().log_prob((x + 1) / 2)


class Beta(nn.Module):
    """
    直接通过alpha和beta参数化的Beta分布类
    用于建模有界连续动作空间（输出动作范围映射至(-1,1)）
    """
    def __init__(self, action_dim):
        """
        参数:
            action_dim: 动作空间维度
        """
        super(Beta, self).__init__()
        self.action_dim = action_dim  # 动作维度

    def forward(self, alpha_beta):
        """
        计算Beta分布的alpha和beta参数
        参数:
            alpha_beta: 输入张量（形状: [batch_size, 2*action_dim]）
                        前半部分用于计算alpha，后半部分用于计算beta
        返回:
            alpha: Beta分布的alpha参数（形状: [batch_size, action_dim]，均>1）
            beta: Beta分布的beta参数（形状: [batch_size, action_dim]，均>1）
        说明:
            1. F.softplus确保输出值非负（softplus(x) = ln(1 + e^x)）
            2. 加1是为了保证alpha和beta > 1，使分布更集中（避免过于平坦）
        """
        # 从输入中提取alpha的参数部分，计算alpha
        alpha = 1 + F.softplus(alpha_beta[:, :self.action_dim])
        # 从输入中提取beta的参数部分，计算beta
        beta  = 1 + F.softplus(alpha_beta[:, self.action_dim:])
        return alpha, beta

    def sample(self, x, deterministic):
        """
        从Beta分布中采样动作（或返回确定性动作）
        参数:
            x: 输入张量（即alpha_beta，形状: [batch_size, 2*action_dim]）
            deterministic: 是否返回确定性动作（True则返回均值，False则采样）
        返回:
            action: 生成的动作（范围(-1,1)，形状: [batch_size, action_dim]）
        """
        if deterministic is False:
            # 非确定性模式：从分布中采样
            action = self.evaluate(x).sample()
        else:
            # 确定性模式：返回分布的均值（alpha/(alpha+beta)）
            return self.evaluate(x).mean

        # 将采样结果从Beta分布的原生范围(0,1)映射到(-1,1)
        return 2 * action - 1

    def evaluate(self, x):
        """
        构建并返回适配(-1,1)范围的Beta分布对象
        参数:
            x: 输入张量（alpha_beta，形状: [batch_size, 2*action_dim]）
        返回:
            BoundedBeta: 适配(-1,1)范围的Beta分布实例
        """
        alpha, beta = self(x)  # 获取alpha和beta参数
        return BoundedBeta(alpha, beta)  # 返回自定义的BoundedBeta分布


# TODO: 为该类想一个更合适的名称
"""通过均值和方差参数化的Beta分布类"""
class Beta2(nn.Module):
    def __init__(self, action_dim, init_std=0.25, learn_std=False):
        """
        参数:
            action_dim: 动作空间维度
            init_std: 初始标准差（用于初始化logstd参数）
            learn_std: 是否将标准差作为可学习参数
        断言:
            Beta分布的最大标准差小于0.5，因此初始标准差必须小于0.5
        """
        super(Beta2, self).__init__()
        assert init_std < 0.5, "Beta分布的最大标准差为0.5"

        self.action_dim = action_dim  # 动作维度

        # 初始化log标准差参数（通过指数转换为实际标准差）
        self.logstd = nn.Parameter(
            torch.ones(1, action_dim) * np.log(init_std),  # 初始值为log(init_std)
            requires_grad=learn_std  # 是否可学习
        )

        self.learn_std = learn_std  # 记录是否学习标准差

    def forward(self, x):
        """
        根据均值和方差计算Beta分布的alpha和beta参数
        参数:
            x: 输入张量（策略网络输出的均值参数，形状: [batch_size, action_dim]）
        返回:
            alpha: Beta分布的alpha参数（形状: [batch_size, action_dim]）
            beta: Beta分布的beta参数（形状: [batch_size, action_dim]）
        说明:
            1. 均值mean通过sigmoid处理，确保在(0,1)范围内（符合Beta分布均值的定义域）
            2. 方差var由logstd指数化后平方得到
            3. alpha和beta通过均值和方差的关系式推导得出（为数值稳定性稍作调整）
        """
        # 计算均值（sigmoid将x映射到(0,1)）
        mean = torch.sigmoid(x) 

        # 计算方差（标准差的平方）
        var = self.logstd.exp().pow(2)

        """
        原始公式（为数值稳定性做了微调）：
        alpha = ((1 - mu) / sigma^2 - 1 / mu) * mu^2
        beta  = alpha * (1 / mu - 1)
        """
        alpha = ((1 - mean) / var) * mean.pow(2) - mean
        beta  = ((1 - mean) / var) * mean - 1 - alpha

        # 问题：如果alpha或beta < 1，可能导致分布特性不佳（注释中标记待优化）

        # 以下为调试相关代码（已注释）
        # assert np.allclose(alpha, ((1 - mean) / var - 1 / mean) * mean.pow(2))
        # assert np.allclose(beta, ((1 - mean) / var - 1 / mean) * mean.pow(2) * (1 / mean - 1))
        # alpha = 1 + F.softplus(alpha)
        # beta  = 1 + F.softplus(beta)
        # print("alpha",alpha)
        # print("beta",beta)
        # print("mu",mean)
        # print("var", var)
        # import pdb
        # pdb.set_trace()

        return alpha, beta

    def sample(self, x, deterministic):
        """
        从Beta分布中采样动作（或返回确定性动作）
        参数:
            x: 输入张量（策略网络输出的均值参数，形状: [batch_size, action_dim]）
            deterministic: 是否返回确定性动作（True则返回均值，False则采样）
        返回:
            action: 生成的动作（范围(-1,1)，形状: [batch_size, action_dim]）
        """
        if deterministic is False:
            # 非确定性模式：从分布中采样
            action = self.evaluate(x).sample()
        else:
            # 确定性模式：返回分布的均值（alpha/(alpha+beta)）
            return self.evaluate(x).mean

        # 将采样结果从Beta分布的原生范围(0,1)映射到(-1,1)
        return 2 * action - 1

    def evaluate(self, x):
        """
        构建并返回适配(-1,1)范围的Beta分布对象
        参数:
            x: 输入张量（策略网络输出的均值参数，形状: [batch_size, action_dim]）
        返回:
            BoundedBeta: 适配(-1,1)范围的Beta分布实例
        """
        alpha, beta = self(x)  # 获取alpha和beta参数
        return BoundedBeta(alpha, beta)  # 返回自定义的BoundedBeta分布