import numpy as np
import torch

# 为单个环境提供向量化接口（将单环境转换为支持批量操作的环境）
class WrapEnv:
    def __init__(self, env_fn):
        # 初始化原始环境（通过环境创建函数）
        self.env = env_fn()

    def __getattr__(self, attr):
        # 属性访问代理：将未定义的属性/方法访问转发给原始环境
        return getattr(self.env, attr)

    def step(self, action):
        """
        执行单步动作并返回批量格式的结果
        参数:
            action: 批量动作数组 (形状: [batch_size, action_dim])
        返回:
            state: 批量状态数组 (形状: [batch_size, state_dim])
            reward: 批量奖励数组 (形状: [batch_size, 1])
            done: 批量终止标志数组 (形状: [batch_size, 1])
            info: 批量额外信息数组 (形状: [batch_size, ...])
        """
        # 从批量动作中提取第一个动作（单环境仅处理一个动作）
        state, reward, done, info = self.env.step(action[0])
        # 将单环境的返回值包装为批量格式（添加batch维度）
        return np.array([state]), np.array([reward]), np.array([done]), np.array([info])

    def render(self):
        # 调用原始环境的渲染方法
        self.env.render()

    def reset(self):
        """重置环境并返回批量格式的初始状态"""
        # 将单环境的初始状态包装为批量格式（添加batch维度）
        return np.array([self.env.reset()])

# TODO: 这个类可能更适合用继承而非包装器实现
# 提供利用环境镜像对称性的接口（生成对称的状态和动作以增强训练）
class SymmetricEnv:    
    def __init__(self, env_fn, mirrored_obs=None, mirrored_act=None, clock_inds=None, obs_fn=None, act_fn=None):
        """
        参数:
            env_fn: 环境创建函数
            mirrored_obs: 观测的镜像索引列表（用于生成镜像观测矩阵）
            mirrored_act: 动作的镜像索引列表（用于生成镜像动作矩阵）
            clock_inds: 观测中时钟信号的索引列表（需特殊处理的周期性信号）
            obs_fn: 自定义观测镜像函数（替代mirrored_obs）
            act_fn: 自定义动作镜像函数（替代mirrored_act）
        """
        # 断言：必须为观测和动作分别提供镜像索引或镜像函数，但不能同时提供
        assert (bool(mirrored_act) ^ bool(act_fn)) and (bool(mirrored_obs) ^ bool(obs_fn)), \
            "You must provide either mirror indices or a mirror function, but not both, for \
             observation and action."

        # 初始化动作镜像机制
        if mirrored_act:
            # 根据镜像索引生成动作镜像矩阵（用于矩阵乘法实现镜像变换）
            self.act_mirror_matrix = torch.Tensor(_get_symmetry_matrix(mirrored_act))
        elif act_fn:
            # 使用自定义动作镜像函数
            assert callable(act_fn), "Action mirror function must be callable"
            self.mirror_action = act_fn

        # 初始化观测镜像机制
        if mirrored_obs:
            # 根据镜像索引生成观测镜像矩阵
            self.obs_mirror_matrix = torch.Tensor(_get_symmetry_matrix(mirrored_obs))
        elif obs_fn:
            # 使用自定义观测镜像函数
            assert callable(obs_fn), "Observation mirror function must be callable"
            self.mirror_observation = obs_fn

        # 记录观测中时钟信号的索引（用于特殊处理周期性信号）
        self.clock_inds = clock_inds
        # 初始化原始环境
        self.env = env_fn()

    def __getattr__(self, attr):
        # 属性访问代理：将未定义的属性/方法访问转发给原始环境
        return getattr(self.env, attr)

    def mirror_action(self, action):
        """使用镜像矩阵对动作进行镜像变换"""
        return action @ self.act_mirror_matrix

    def mirror_observation(self, obs):
        """使用镜像矩阵对观测进行镜像变换"""
        return obs @ self.obs_mirror_matrix

    # 当观测中包含时钟信号时使用。此时，创建SymmetricEnv时输入的mirrored_obs向量不应移动时钟输入顺序。
    # 需要输入观测向量中时钟所在的索引。
    def mirror_clock_observation(self, obs):
        """处理包含时钟信号的观测镜像（特殊处理周期性信号）"""
        # print("obs.shape = ", obs.shape)
        # print("obs_mirror_matrix.shape = ", self.obs_mirror_matrix.shape)
        
        # 创建与输入观测相同形状的零矩阵，用于存储镜像后的观测
        mirror_obs_batch = torch.zeros_like(obs)
        # 固定历史状态长度为1（当前仅支持单步历史）
        history_len = 1 
        
        for block in range(history_len):
            # 从观测中提取当前历史块
            obs_ = obs[:, self.base_obs_len*block : self.base_obs_len*(block+1)]
            # 应用常规镜像变换
            mirror_obs = obs_ @ self.obs_mirror_matrix
            # 提取时钟信号部分
            clock = mirror_obs[:, self.clock_inds]
            
            # 对每个时钟信号进行特殊处理（保持周期性但调整相位）
            for i in range(np.shape(clock)[1]):
                # 通过arcsin提取相位，加π后再用sin重构，实现相位反转
                mirror_obs[:, self.clock_inds[i]] = np.sin(np.arcsin(clock[:, i]) + np.pi)
            
            # 将处理后的镜像观测块存入结果矩阵
            mirror_obs_batch[:, self.base_obs_len*block : self.base_obs_len*(block+1)] = mirror_obs
        
        return mirror_obs_batch

# 辅助函数：根据镜像索引列表生成对称变换矩阵
def _get_symmetry_matrix(mirrored):
    """
    根据镜像索引生成对称变换矩阵
    参数:
        mirrored: 镜像索引列表（每个元素表示原索引到目标索引的映射，负值表示取反）
    返回:
        mat: 对称变换矩阵 (形状: [len(mirrored), len(mirrored)])
    """
    # 获取维度数量
    numel = len(mirrored)
    # 初始化零矩阵
    mat = np.zeros((numel, numel))

    # 填充矩阵：将原索引i映射到目标索引j，并根据符号确定是否取反
    for (i, j) in zip(np.arange(numel), np.abs(np.array(mirrored).astype(int))):
        mat[i, j] = np.sign(mirrored[i])  # sign确定映射关系中的符号（1或-1）

    return mat