import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import sqrt

# 从自定义模块导入基础网络类
from rl.policies.base import Net

class Actor(Net):
    """
    演员网络基类，继承自基础网络类Net
    所有具体的演员网络都需要继承此类并实现forward方法
    """
    def __init__(self):
        super(Actor, self).__init__()

    def forward(self):
        """
        前向传播方法，必须在子类中实现
        用于定义网络的计算流程
        """
        raise NotImplementedError  # 抛出未实现错误，强制子类重写

class Linear_Actor(Actor):
    """
    线性演员网络（无激活函数的简单网络）
    仅包含两层线性层，用于输出动作
    """
    def __init__(self, state_dim, action_dim, hidden_size=32):
        """
        初始化线性演员网络
        参数：
            state_dim: 状态空间维度
            action_dim: 动作空间维度
            hidden_size: 隐藏层大小，默认32
        """
        super(Linear_Actor, self).__init__()

        # 定义两层线性层
        self.l1 = nn.Linear(state_dim, hidden_size)  # 输入层到隐藏层
        self.l2 = nn.Linear(hidden_size, action_dim)  # 隐藏层到输出层

        self.action_dim = action_dim  # 保存动作维度

        # 初始化所有参数为0
        for p in self.parameters():
            p.data = torch.zeros(p.shape)

    def forward(self, state):
        """
        前向传播计算
        参数：
            state: 输入状态
        返回：
            网络输出的动作
        """
        a = self.l1(state)  # 第一层线性变换
        a = self.l2(a)      # 第二层线性变换
        return a

class FF_Actor(Actor):
    """
    前馈演员网络（Feed-Forward Actor）
    包含多层全连接网络，使用指定的非线性激活函数
    """
    def __init__(self, state_dim, action_dim, layers=(256, 256), nonlinearity=F.relu):
        """
        初始化前馈演员网络
        参数：
            state_dim: 状态空间维度
            action_dim: 动作空间维度
            layers: 各隐藏层大小的元组，默认(256, 256)
            nonlinearity: 非线性激活函数，默认ReLU
        """
        super(FF_Actor, self).__init__()

        # 定义网络层列表
        self.actor_layers = nn.ModuleList()
        self.actor_layers += [nn.Linear(state_dim, layers[0])]  # 输入层到第一层隐藏层
        # 添加后续隐藏层
        for i in range(len(layers)-1):
            self.actor_layers += [nn.Linear(layers[i], layers[i+1])]
        # 输出层（动作维度）
        self.network_out = nn.Linear(layers[-1], action_dim)

        self.action_dim = action_dim  # 动作维度
        self.nonlinearity = nonlinearity  # 激活函数

        self.initialize_parameters()  # 初始化参数

    def forward(self, state, deterministic=True):
        """
        前向传播计算
        参数：
            state: 输入状态
            deterministic: 是否确定性输出（本实现中未实际使用）
        返回：
            经过tanh激活的动作（范围[-1,1]）
        """
        x = state
        # 逐层计算，应用激活函数
        for idx, layer in enumerate(self.actor_layers):
            x = self.nonlinearity(layer(x))

        # 输出层通过tanh激活，将动作限制在[-1,1]
        action = torch.tanh(self.network_out(x))
        return action


class LSTM_Actor(Actor):
    """
    LSTM演员网络（长短期记忆网络）
    适用于处理序列数据，能保留时序信息
    """
    def __init__(self, state_dim, action_dim, layers=(128, 128), nonlinearity=torch.tanh):
        """
        初始化LSTM演员网络
        参数：
            state_dim: 状态空间维度
            action_dim: 动作空间维度
            layers: 各LSTM层隐藏状态大小的元组，默认(128, 128)
            nonlinearity: 非线性激活函数，默认tanh
        """
        super(LSTM_Actor, self).__init__()

        # 定义LSTM层列表
        self.actor_layers = nn.ModuleList()
        self.actor_layers += [nn.LSTMCell(state_dim, layers[0])]  # 第一层LSTM
        # 添加后续LSTM层
        for i in range(len(layers)-1):
            self.actor_layers += [nn.LSTMCell(layers[i], layers[i+1])]
        # 输出层（注意：原代码此处索引可能有误，应为layers[-1]）
        self.network_out = nn.Linear(layers[i-1], action_dim)

        self.action_dim = action_dim  # 动作维度
        self.init_hidden_state()  # 初始化隐藏状态
        self.nonlinearity = nonlinearity  # 激活函数

    def get_hidden_state(self):
        """获取当前的隐藏状态和细胞状态"""
        return self.hidden, self.cells

    def set_hidden_state(self, data):
        """
        设置隐藏状态和细胞状态
        参数：
            data: 包含隐藏状态和细胞状态的元组
        """
        if len(data) != 2:
            print("获取到无效的隐藏状态数据。")
            exit(1)  # 退出程序

        self.hidden, self.cells = data

    def init_hidden_state(self, batch_size=1):
        """
        初始化隐藏状态和细胞状态为全零
        参数：
            batch_size: 批次大小，默认1
        """
        self.hidden = [torch.zeros(batch_size, l.hidden_size) for l in self.actor_layers]
        self.cells = [torch.zeros(batch_size, l.hidden_size) for l in self.actor_layers]

    def forward(self, x, deterministic=True):
        """
        前向传播计算
        参数：
            x: 输入状态（可能是单步或序列）
            deterministic: 是否确定性输出（本实现中未实际使用）
        返回：
            网络输出的动作
        """
        dims = len(x.size())  # 获取输入维度

        if dims == 3:  # 如果输入是轨迹批次 (时间步, 批次, 特征)
            self.init_hidden_state(batch_size=x.size(1))  # 初始化隐藏状态
            y = []
            # 按时间步处理
            for t, x_t in enumerate(x):
                for idx, layer in enumerate(self.actor_layers):
                    c, h = self.cells[idx], self.hidden[idx]
                    # LSTM计算（输入，(隐藏状态, 细胞状态)）
                    self.hidden[idx], self.cells[idx] = layer(x_t, (h, c))
                    x_t = self.hidden[idx]  # 更新当前时间步的输出
                y.append(x_t)
            x = torch.stack([x_t for x_t in y])  # 堆叠所有时间步的输出

        else:
            if dims == 1:  # 如果是单个时间步的状态（无批次维度）
                x = x.view(1, -1)  # 增加批次维度

            # 处理单步输入
            for idx, layer in enumerate(self.actor_layers):
                h, c = self.hidden[idx], self.cells[idx]
                self.hidden[idx], self.cells[idx] = layer(x, (h, c))
                x = self.hidden[idx]
            x = self.nonlinearity(self.network_out(x))  # 应用激活函数

            if dims == 1:  # 恢复原始维度
                x = x.view(-1)

        action = self.network_out(x)  # 输出动作
        return action


class Gaussian_FF_Actor(Actor):  # 与其他演员命名约定更一致
    """
    高斯前馈演员网络
    输出动作的高斯分布参数（均值和标准差），适用于随机策略
    """
    def __init__(self, state_dim, action_dim, layers=(256, 256), nonlinearity=torch.nn.functional.relu,
                 init_std=0.2, learn_std=False, bounded=False, normc_init=True):
        """
        初始化高斯前馈演员网络
        参数：
            state_dim: 状态空间维度
            action_dim: 动作空间维度
            layers: 各隐藏层大小的元组，默认(256, 256)
            nonlinearity: 非线性激活函数，默认ReLU
            init_std: 初始标准差，默认0.2
            learn_std: 是否学习标准差，默认False
            bounded: 动作是否有界（是否使用tanh激活），默认False
            normc_init: 是否使用normc参数初始化，默认True
        """
        super(Gaussian_FF_Actor, self).__init__()

        # 定义网络层列表
        self.actor_layers = nn.ModuleList()
        self.actor_layers += [nn.Linear(state_dim, layers[0])]  # 输入层到第一层隐藏层
        # 添加后续隐藏层
        for i in range(len(layers)-1):
            self.actor_layers += [nn.Linear(layers[i], layers[i+1])]
        # 输出动作均值的层
        self.means = nn.Linear(layers[-1], action_dim)

        self.learn_std = learn_std  # 是否学习标准差
        if self.learn_std:
            # 标准差作为可学习参数
            self.stds = nn.Parameter(init_std * torch.ones(action_dim))
        else:
            # 标准差为固定值
            self.stds = init_std * torch.ones(action_dim)

        self.action_dim = action_dim  # 动作维度
        self.state_dim = state_dim    # 状态维度
        self.nonlinearity = nonlinearity  # 激活函数

        # 初始化输入归一化参数（初始为无归一化）
        self.obs_std = 1.0    # 观测标准差
        self.obs_mean = 0.0   # 观测均值

        self.normc_init = normc_init  # 是否使用normc初始化

        self.bounded = bounded  # 动作是否有界

        self.init_parameters()  # 初始化参数

    def init_parameters(self):
        """初始化网络参数"""
        if self.normc_init:
            self.apply(normc_fn)  # 应用normc初始化
            self.means.weight.data.mul_(0.01)  # 缩放均值输出层的权重

    def _get_dist_params(self, state):
        """
        获取高斯分布的参数（均值和标准差）
        参数：
            state: 输入状态
        返回：
            mean: 分布均值
            sd: 分布标准差
        """
        # 对输入状态进行归一化
        state = (state - self.obs_mean) / self.obs_std

        x = state
        # 逐层计算
        for l in self.actor_layers:
            x = self.nonlinearity(l(x))
        mean = self.means(x)  # 计算均值

        if self.bounded:
            mean = torch.tanh(mean)  # 若动作有界，用tanh限制范围

        sd = torch.zeros_like(mean)
        if hasattr(self, 'stds'):
            sd = self.stds  # 获取标准差
        return mean, sd

    def forward(self, state, deterministic=True):
        """
        前向传播计算
        参数：
            state: 输入状态
            deterministic: 是否确定性输出（True则返回均值，False则采样）
        返回：
            动作（确定性或采样得到）
        """
        mu, sd = self._get_dist_params(state)  # 获取分布参数

        if not deterministic:
            # 从高斯分布中采样动作
            action = torch.distributions.Normal(mu, sd).sample()
        else:
            # 确定性输出（使用均值）
            action = mu

        return action

    def distribution(self, inputs):
        """
        获取输入对应的高斯分布
        参数：
            inputs: 输入状态
        返回：
            高斯分布对象
        """
        mu, sd = self._get_dist_params(inputs)
        return torch.distributions.Normal(mu, sd)


class Gaussian_LSTM_Actor(Actor):
    """
    高斯LSTM演员网络
    结合LSTM的时序处理能力和高斯分布的随机策略
    """
    def __init__(self, state_dim, action_dim, layers=(128, 128), nonlinearity=F.tanh, normc_init=False,
                 init_std=0.2, learn_std=False):
        """
        初始化高斯LSTM演员网络
        参数：
            state_dim: 状态空间维度
            action_dim: 动作空间维度
            layers: 各LSTM层隐藏状态大小的元组，默认(128, 128)
            nonlinearity: 非线性激活函数，默认tanh
            normc_init: 是否使用normc参数初始化，默认False
            init_std: 初始标准差，默认0.2
            learn_std: 是否学习标准差，默认False
        """
        super(Gaussian_LSTM_Actor, self).__init__()

        # 定义LSTM层列表
        self.actor_layers = nn.ModuleList()
        self.actor_layers += [nn.LSTMCell(state_dim, layers[0])]  # 第一层LSTM
        # 添加后续LSTM层
        for i in range(len(layers)-1):
            self.actor_layers += [nn.LSTMCell(layers[i], layers[i+1])]
        # 输出层（注意：原代码此处索引可能有误，应为layers[-1]）
        self.network_out = nn.Linear(layers[i-1], action_dim)

        self.action_dim = action_dim  # 动作维度
        self.state_dim = state_dim    # 状态维度
        self.init_hidden_state()      # 初始化隐藏状态
        self.nonlinearity = nonlinearity  # 激活函数

        # 初始化输入归一化参数（初始为无归一化）
        self.obs_std = 1.0    # 观测标准差
        self.obs_mean = 0.0   # 观测均值

        self.learn_std = learn_std  # 是否学习标准差
        if self.learn_std:
            # 标准差作为可学习参数
            self.stds = nn.Parameter(init_std * torch.ones(action_dim))
        else:
            # 标准差为固定值
            self.stds = init_std * torch.ones(action_dim)

        if normc_init:
            self.initialize_parameters()  # 初始化参数

        self.act = self.forward  # 动作函数别名

    def _get_dist_params(self, state):
        """
        获取高斯分布的参数（均值和标准差）
        参数：
            state: 输入状态（可能是单步或序列）
        返回：
            mu: 分布均值
            sd: 分布标准差
        """
        # 对输入状态进行归一化
        state = (state - self.obs_mean) / self.obs_std

        dims = len(state.size())  # 获取输入维度

        x = state
        if dims == 3:  # 如果输入是轨迹批次 (时间步, 批次, 特征)
            self.init_hidden_state(batch_size=x.size(1))  # 初始化隐藏状态
            action = []
            y = []
            # 按时间步处理
            for t, x_t in enumerate(x):
                for idx, layer in enumerate(self.actor_layers):
                    c, h = self.cells[idx], self.hidden[idx]
                    # LSTM计算
                    self.hidden[idx], self.cells[idx] = layer(x_t, (h, c))
                    x_t = self.hidden[idx]  # 更新当前时间步的输出
                y.append(x_t)
            x = torch.stack([x_t for x_t in y])  # 堆叠所有时间步的输出

        else:
            if dims == 1:  # 如果是单个时间步的状态（无批次维度）
                x = x.view(1, -1)  # 增加批次维度

            # 处理单步输入
            for idx, layer in enumerate(self.actor_layers):
                h, c = self.hidden[idx], self.cells[idx]
                self.hidden[idx], self.cells[idx] = layer(x, (h, c))
                x = self.hidden[idx]

            if dims == 1:  # 恢复原始维度
                x = x.view(-1)

        mu = self.network_out(x)  # 计算均值
        sd = self.stds            # 获取标准差
        return mu, sd

    def init_hidden_state(self, batch_size=1):
        """
        初始化隐藏状态和细胞状态为全零
        参数：
            batch_size: 批次大小，默认1
        """
        self.hidden = [torch.zeros(batch_size, l.hidden_size) for l in self.actor_layers]
        self.cells = [torch.zeros(batch_size, l.hidden_size) for l in self.actor_layers]

    def forward(self, state, deterministic=True):
        """
        前向传播计算
        参数：
            state: 输入状态
            deterministic: 是否确定性输出（True则返回均值，False则采样）
        返回：
            动作（确定性或采样得到）
        """
        mu, sd = self._get_dist_params(state)  # 获取分布参数

        if not deterministic:
            # 从高斯分布中采样动作
            action = torch.distributions.Normal(mu, sd).sample()
        else:
            # 确定性输出（使用均值）
            action = mu

        return action

    def distribution(self, inputs):
        """
        获取输入对应的高斯分布
        参数：
            inputs: 输入状态
        返回：
            高斯分布对象
        """
        mu, sd = self._get_dist_params(inputs)
        return torch.distributions.Normal(mu, sd)

# 高斯MLP的初始化方案（来自PPO论文）
# 注意：函数名与参数名相同曾导致严重bug
# 显然在Python中，"if <函数名>"会被评估为True...
def normc_fn(m):
    classname = m.__class__.__name__
    # 如果是线性层
    if classname.find('Linear') != -1:
        m.weight.data.normal_(0, 1)  # 权重初始化为均值0、标准差1的正态分布
        # 归一化权重（使得每行的L2范数为1）
        m.weight.data *= 1 / torch.sqrt(m.weight.data.pow(2).sum(1, keepdim=True))
        # 偏置初始化为0
        if m.bias is not None:
            m.bias.data.fill_(0)
