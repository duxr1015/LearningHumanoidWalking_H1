# 改编自 https://github.com/openai/baselines/blob/master/baselines/common/vec_env/vec_normalize.py
# 感谢原作者及OpenAI提供的代码

import numpy as np
import functools
import torch
import ray  # 用于并行计算的框架

from .wrappers import WrapEnv  # 导入环境包装类


@ray.remote  # 标记为ray远程函数，支持并行执行
def _run_random_actions(iter, policy, env_fn, noise_std):
    """
    并行执行随机动作以收集环境状态数据，用于计算归一化参数
    参数:
        iter: 执行的步数
        policy: 策略函数（用于生成基础动作）
        env_fn: 环境创建函数
        noise_std: 高斯噪声的标准差（用于在基础动作上添加探索噪声）
    返回:
        states: 收集到的状态数据（形状: [iter, state_dim]）
    """
    # 用WrapEnv包装环境，使其支持批量接口
    env = WrapEnv(env_fn)
    # 初始化状态存储数组（行数为步数，列数为状态维度）
    states = np.zeros((iter, env.observation_space.shape[0]))

    # 重置环境获取初始状态
    state = env.reset()
    for t in range(iter):
        # 存储当前状态
        states[t, :] = state

        # 将状态转换为Tensor输入策略
        state = torch.Tensor(state)

        # 用策略生成基础动作
        action = policy(state)

        # 在基础动作上添加高斯噪声（增强探索，收集更多样的状态）
        action = action + torch.randn(action.size()) * noise_std

        # 执行带噪声的动作，获取新状态
        state, _, done, _ = env.step(action.data.numpy())

        # 如果环境终止，重置环境
        if done:
            state = env.reset()
    
    return states


def get_normalization_params(iter, policy, env_fn, noise_std, procs=4):
    """
    并行收集状态数据并计算归一化参数（均值和标准差）
    参数:
        iter: 总收集步数
        policy: 策略函数
        env_fn: 环境创建函数
        noise_std: 动作噪声标准差
        procs: 并行进程数
    返回:
        mean: 状态均值（形状: [state_dim]）
        std: 状态标准差（形状: [state_dim]，加1e-8避免除以0）
    """
    print(f"使用{iter}步收集输入归一化数据，噪声标准差={noise_std}...")

    # 启动多个并行任务（每个任务处理iter//procs步）
    states_ids = [_run_random_actions.remote(iter // procs, policy, env_fn, noise_std) for _ in range(procs)]

    # 收集所有并行任务的结果
    states = []
    for _ in range(procs):
        # 等待一个任务完成
        ready_ids, _ = ray.wait(states_ids, num_returns=1)
        # 获取结果并添加到列表
        states.extend(ray.get(ready_ids[0]))
        # 移除已完成的任务ID
        states_ids.remove(ready_ids[0])

    print("输入归一化数据收集完成。")

    # 计算所有状态的均值和标准差（加1e-8避免方差为0时除零错误）
    return np.mean(states, axis=0), np.sqrt(np.var(states, axis=0) + 1e-8)


# 返回一个函数，该函数创建归一化环境，并使用带噪声的确定性策略预收集数据以初始化归一化参数
def PreNormalizer(iter, noise_std, policy, *args, **kwargs):
    """
    预归一化器生成函数
    参数:
        iter: 预收集数据的步数
        noise_std: 动作噪声标准差
        policy: 用于生成基础动作的策略
        *args, **kwargs: 传递给Normalize类的其他参数
    返回:
        _Normalizer: 创建并初始化预归一化环境的函数
    """

    # 禁用梯度计算（仅用于数据收集，不更新策略）
    @torch.no_grad()
    def pre_normalize(env, policy, num_iter, noise_std):
        """在环境中执行带噪声的动作，收集数据并更新归一化统计量"""
        # 保存环境原有的在线更新设置（后续恢复）
        online_val = env.online
        # 开启在线更新（允许在收集数据时更新均值和方差）
        env.online = True

        # 重置环境获取初始状态
        state = env.reset()

        for t in range(num_iter):
            # 将状态转换为Tensor
            state = torch.Tensor(state)

            # 用策略生成动作（假设策略返回(值, 动作)，此处取动作）
            _, action = policy(state)

            # 在动作上添加高斯噪声（增强探索）
            action = action + torch.randn(action.size()) * noise_std

            # 执行动作，获取新状态
            state, _, done, _ = env.step(action.data.numpy())

            # 若环境终止，重置环境
            if done:
                state = env.reset()

        # 恢复环境原有的在线更新设置
        env.online = online_val
    
    def _Normalizer(venv):
        """创建归一化环境并执行预收集数据"""
        # 用Normalize类包装环境
        venv = Normalize(venv, *args, **kwargs)

        print(f"使用{iter}步收集输入归一化数据，噪声标准差={noise_std}...")
        # 执行预收集数据并更新归一化统计量
        pre_normalize(venv, policy, iter, noise_std)
        print("输入归一化数据收集完成。")

        return venv

    return _Normalizer


# 返回一个函数，该函数创建归一化环境（不进行预收集数据，仅初始化）
def Normalizer(*args, **kwargs):
    """
    归一化器生成函数
    参数:
        *args, **kwargs: 传递给Normalize类的参数
    返回:
        _Normalizer: 创建归一化环境的函数
    """
    def _Normalizer(venv):
        return Normalize(venv, *args, **kwargs)

    return _Normalizer


class Normalize:
    """
    向量环境的归一化包装类，用于标准化观测和奖励，提升训练稳定性
    """
    def __init__(self, 
                 venv,
                 ob_rms=None,  # 观测的均值方差统计器（RunningMeanStd实例）
                 ob=True,  # 是否归一化观测
                 ret=False,  # 是否归一化奖励
                 clipob=10.,  # 观测的剪辑范围（[-clipob, clipob]）
                 cliprew=10.,  # 奖励的剪辑范围（[-cliprew, cliprew]）
                 online=True,  # 是否在线更新均值方差
                 gamma=1.0,  # 奖励折扣因子（用于计算累积回报）
                 epsilon=1e-8):  # 数值稳定性常数（避免除零）

        self.venv = venv  # 被包装的向量环境
        self._observation_space = venv.observation_space  # 观测空间（继承自原环境）
        self._action_space = venv.action_space  # 动作空间（继承自原环境）

        # 初始化观测的均值方差统计器（若未提供则新建）
        if ob_rms is not None:
            self.ob_rms = ob_rms
        else:
            self.ob_rms = RunningMeanStd(shape=self._observation_space.shape) if ob else None

        # 初始化奖励的均值方差统计器（若启用奖励归一化）
        self.ret_rms = RunningMeanStd(shape=()) if ret else None
        self.clipob = clipob  # 观测剪辑阈值
        self.cliprew = cliprew  # 奖励剪辑阈值
        self.ret = np.zeros(self.num_envs)  # 存储每个环境的累积回报
        self.gamma = gamma  # 折扣因子
        self.epsilon = epsilon  # 数值稳定性常数

        self.online = online  # 在线更新开关（True则实时更新均值方差）

    def step(self, vac):
        """
        执行一步动作并返回归一化后的结果
        参数:
            vac: 批量动作（形状: [num_envs, action_dim]）
        返回:
            obs: 归一化并剪辑后的观测
            rews: 归一化并剪辑后的奖励
            news: 终止标志数组
            infos: 额外信息数组
        """
        # 调用原环境的step方法执行动作
        obs, rews, news, infos = self.venv.step(vac)

        # 更新累积回报（用于奖励归一化）
        # self.ret = self.ret * self.gamma + rews
        # 对观测进行归一化和剪辑
        obs = self._obfilt(obs)

        # 若启用奖励归一化
        if self.ret_rms: 
            # 若开启在线更新，用累积回报更新奖励的均值方差
            if self.online:
                self.ret_rms.update(self.ret)
            
            # 奖励归一化（除以标准差）并剪辑
            rews = np.clip(rews / np.sqrt(self.ret_rms.var + self.epsilon), -self.cliprew, self.cliprew)

        return obs, rews, news, infos

    def _obfilt(self, obs):
        """
        对观测进行归一化和剪辑
        参数:
            obs: 原始观测（形状: [num_envs, state_dim]）
        返回:
            归一化并剪辑后的观测
        """
        if self.ob_rms:  # 若启用观测归一化
            # 若开启在线更新，用当前观测更新均值方差
            if self.online:
                self.ob_rms.update(obs)
            
            # 观测归一化（(观测 - 均值) / 标准差）并剪辑
            obs = np.clip((obs - self.ob_rms.mean) / np.sqrt(self.ob_rms.var + self.epsilon), -self.clipob, self.clipob)
            return obs
        else:
            # 不启用归一化则直接返回原始观测
            return obs

    def reset(self):
        """重置所有环境并返回归一化后的初始观测"""
        obs = self.venv.reset()
        return self._obfilt(obs)

    @property
    def action_space(self):
        """返回动作空间（继承自原环境）"""
        return self._action_space

    @property
    def observation_space(self):
        """返回观测空间（继承自原环境）"""
        return self._observation_space

    def close(self):
        """关闭环境"""
        self.venv.close()
    
    def render(self):
        """渲染环境"""
        self.venv.render()

    @property
    def num_envs(self):
        """返回环境数量（继承自原环境）"""
        return self.venv.num_envs


class RunningMeanStd(object):
    """
    用于增量计算均值和方差的类（支持并行更新）
    参考: https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
    """
    def __init__(self, epsilon=1e-4, shape=()):
        """
        参数:
            epsilon: 初始计数（避免除以零）
            shape: 均值和方差的形状（与观测/奖励维度匹配）
        """
        self.mean = np.zeros(shape, 'float64')  # 均值
        self.var = np.zeros(shape, 'float64')   # 方差
        self.count = epsilon  # 计数（初始为epsilon避免除零）

    def update(self, x):
        """
        用新批次数据更新均值和方差
        参数:
            x: 新批次数据（形状: [batch_size, ...]，与初始化时的shape匹配）
        """
        # 计算批次数据的均值和方差
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]  # 批次大小

        # 计算新旧均值的差
        delta = batch_mean - self.mean
        # 总计数（原有计数 + 新批次计数）
        tot_count = self.count + batch_count

        # 更新均值：新均值 = 旧均值 + （批次均值 - 旧均值）* 批次计数 / 总计数
        new_mean = self.mean + delta * batch_count / tot_count        
        # 计算合并后的平方和（用于更新方差）
        m_a = self.var * (self.count)  # 原有数据的平方和
        m_b = batch_var * (batch_count)  # 新批次数据的平方和
        # 合并平方和（包含新旧均值差的校正项）
        M2 = m_a + m_b + np.square(delta) * self.count * batch_count / (self.count + batch_count)
        # 更新方差：新方差 = 合并平方和 / 总计数
        new_var = M2 / (self.count + batch_count)

        # 更新计数
        new_count = batch_count + self.count

        # 保存更新后的值
        self.mean = new_mean
        self.var = new_var
        self.count = new_count        


def test_runningmeanstd():
    """测试RunningMeanStd类的正确性（验证增量计算与直接计算结果一致）"""
    # 测试不同形状的数据（1D、2D）
    for (x1, x2, x3) in [
        (np.random.randn(3), np.random.randn(4), np.random.randn(5)),  # 1D数据
        (np.random.randn(3,2), np.random.randn(4,2), np.random.randn(5,2)),  # 2D数据
        ]:

        # 初始化RunningMeanStd（形状与输入数据的最后一维匹配）
        rms = RunningMeanStd(epsilon=0.0, shape=x1.shape[1:])

        # 合并所有数据直接计算均值和方差
        x = np.concatenate([x1, x2, x3], axis=0)
        ms1 = [x.mean(axis=0), x.var(axis=0)]
        # 增量更新RunningMeanStd
        rms.update(x1)
        rms.update(x2)
        rms.update(x3)
        ms2 = [rms.mean, rms.var]

        # 验证增量计算结果与直接计算结果一致
        assert np.allclose(ms1, ms2)