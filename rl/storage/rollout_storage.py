import torch

class PPOBuffer:
    def __init__(self,obs_len=1, act_len=1, gamma=0.99, lam=0.95, use_gae=False, size=1):
        # 初始化存储缓冲区(张量)
        self.states = torch.zeros(size, obs_len, dtype=float)  # 状态存储（size：最大容量，obs_len：状态维度）
        self.actions = torch.zeros(size, act_len, dtype=float) # 动作存储（act_len：动作维度）
        self.rewards = torch.zeros(size, 1, dtype=float)        # 奖励存储
        self.values  = torch.zeros(size, 1, dtype=float)        # 状态价值存储（Critic输出）
        self.returns = torch.zeros(size, 1, dtype=float)        # 累积回报存储（用于计算损失）
        self.dones   = torch.zeros(size, 1, dtype=float)        # 终止标志存储（1表示episode结束）

        # 超参数
        self.gamma = gamma  # 折扣因子（未来奖励的衰减系数）
        self.lam = lam      # GAE（广义优势估计）的平滑系数
        self.use_gae = use_gae  # 是否使用GAE计算优势（代码中未直接体现，预留扩展）

        # 缓冲区指针与轨迹索引
        self.ptr = 0  # 当前存储位置指针（指向下次存储的索引）
        self.traj_idx = [0]  # 轨迹分割索引（记录每个episode的起始位置）

    def __len__(self):

        return self.ptr  # 返回当前存储的样本数量（指针位置）
    
    def store(self,state,action,reward,value,done):
        """
        存储单个时间步的状态、动作、奖励、价值和终止标志
        """
        self.states[self.ptr] = state  # 存储当前状态（如机器人关节角度、位置等）
        self.actions[self.ptr] = action  # 存储当前动作（如关节力矩、速度等）
        self.rewards[self.ptr] = reward  # 存储当前奖励（如前进速度奖励、平衡奖励等）
        self.values[self.ptr] = value    # 存储Critic估计的状态价值V(s)
        self.dones[self.ptr] = done      # 存储终止标志（1表示当前步结束episode）
        self.ptr += 1  # 更新指针位置，准备存储下一个样本

    
    def finish_path(self,last_val=None):
        """
        当一个episode结束时，计算该轨迹的累积回报（returns）
        last_val：最后一个状态的价值估计（用于处理未终止的轨迹）
        """

        # 记录当前轨迹的结束位置（更新轨迹索引）
        self.traj_idx += [self.ptr]  # traj_idx格式：[episode1_start, episode1_end, episode2_start, ...]
        
        # 提取当前轨迹的奖励（从上个轨迹结束位置到当前指针）
        rewards = self.rewards[self.traj_idx[-2]:self.traj_idx[-1], 0]  # 切片获取当前episode的奖励序列

        # 初始化累积回报R（最后一个状态的价值，若episode终止则为0）
        R = last_val.squeeze(0) if last_val is not None else 0.0  # squeeze(0)移除批量维度

        # 倒序计算累积回报（从最后一步到第一步）
        returns = torch.zeros_like(rewards)  # 存储每个时间步的累积回报

        for i in range(len(rewards) - 1, -1, -1):
            # 累积回报公式：R_t = r_t + gamma * R_{t+1}
            # 若当前步是终止状态（done=1），则R_{t+1}=0
            R = rewards[i] + self.gamma * R * (1 - self.dones[self.traj_idx[-2] + i])
            returns[i] = R  # 存储计算结果

        # 将计算好的累积回报写入缓冲区
        self.returns[self.traj_idx[-2]:self.traj_idx[-1], 0] = returns

        # 标记当前轨迹的最后一步为终止（方便后续处理）
        self.dones[-1] = True

    def get_data(self):
        """返回所有存储的轨迹数据，并隐含重置缓冲区（通过返回切片实现）"""
        # 计算每个episode的长度（结束索引 - 开始索引）
        ep_lens = [j - i for i, j in zip(self.traj_idx, self.traj_idx[1:])]
        
        # 计算每个episode的总奖励（累积该episode的所有奖励）
        ep_rewards = [
            float(sum(self.rewards[int(i):int(j)])) for i, j in zip(self.traj_idx, self.traj_idx[1:])
        ]
        
        # 打包所有数据为字典
        data = {
            'states': self.states[:self.ptr],        # 所有状态（切片到当前指针，忽略未使用空间）
            'actions': self.actions[:self.ptr],      # 所有动作
            'rewards': self.rewards[:self.ptr],      # 所有奖励
            'values': self.values[:self.ptr],        # 所有状态价值
            'returns': self.returns[:self.ptr],      # 所有累积回报
            'dones': self.dones[:self.ptr],          # 所有终止标志
            'traj_idx': torch.Tensor(self.traj_idx), # 轨迹分割索引（用于区分不同episode）
            'ep_lens': torch.Tensor(ep_lens),        # 每个episode的长度
            'ep_rewards': torch.Tensor(ep_rewards),  # 每个episode的总奖励
        }
        return data

