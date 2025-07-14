import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import sqrt



# 权重初始化函数
def normc_fn(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.data.normal_(0, 1)
        m.weight.data *= 1 / torch.sqrt(m.weight.data.pow(2).sum(1,keepdim=True))
        if m.bias is not None:
            m.bias.data.fill_(0)

# 策略网络基类
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # Welford 算法参数：用于在线计算状态的均值和方差(滚动更新)
        self.welford_state_mean = torch.zeros(1)  #状态均值
        self.welford_state_mean_diff = torch.ones(1)  #状态平方差累积
        self.welford_state_n = 1  #样本数量
    
    def forward(self):
        raise NotImplementedError("This method should be overridden by subclasses.") 
    
    def normalize_state(self, state, update=True):
        state = torch.Tensor(state)  # 转换为Tensor

        # 初始化：首次调用时根据状态维度设置均值和方差的初始值
        if self.welford_state_n == 1:
            self.welford_state_mean = torch.zeros(state.size(-1))
            self.welford_state_mean_diff = torch.ones(state.size(-1))

        # 更新统计量（仅训练阶段启用）
        if update:
            # 处理单样本、批量样本、序列样本等不同输入格式
            if len(state.size()) == 1:  # 单样本
                state_old = self.welford_state_mean
                self.welford_state_mean += (state - state_old) / self.welford_state_n
                self.welford_state_mean_diff += (state - state_old) * (state - self.welford_state_mean)
                self.welford_state_n += 1
            elif len(state.size()) == 2:  # 批量样本 (batch_size, state_dim)
                for s in state:  # 遍历批次中的每个样本
                    state_old = self.welford_state_mean
                    self.welford_state_mean += (s - state_old) / self.welford_state_n
                    self.welford_state_mean_diff += (s - state_old) * (s - self.welford_state_mean)
                self.welford_state_n += state.shape[0]  # 增加样本计数
            elif len(state.size()) == 3:  # 序列样本 (batch_size, seq_len, state_dim)
                for seq in state:  # 遍历每个序列
                    for s in seq:  # 遍历序列中的每个时间步
                        state_old = self.welford_state_mean
                        self.welford_state_mean += (s - state_old) / self.welford_state_n
                        self.welford_state_mean_diff += (s - state_old) * (s - self.welford_state_mean)
                self.welford_state_n += state.shape[0] * state.shape[1]  # 增加样本计数

        # 归一化：(状态 - 均值) / 标准差（标准差由方差开方得到）
        return (state - self.welford_state_mean) / torch.sqrt(self.welford_state_mean_diff / self.welford_state_n)

    def copy_normalizer_stats(self, net):
        self.welford_state_mean = net.self_state_mean
        self.welford_state_mean_diff = net.welford_state_mean_diff
        self.welford_state_n = net.welford_state_n
  
    def initialize_parameters(self):
        self.apply(normc_fn)

    