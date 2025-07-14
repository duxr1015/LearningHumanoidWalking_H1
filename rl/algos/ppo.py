"""近端策略优化（PPO，clip目标版本）"""
from copy import deepcopy

import torch
import torch.optim as optim
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from torch.nn.utils.rnn import pad_sequence  # 用于填充变长序列
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter  # 用于训练日志可视化

from pathlib import Path  # 路径处理
import sys
import time
import numpy as np
import datetime

import ray  # 并行计算框架

from rl.storage.rollout_storage import PPOBuffer  # 轨迹数据缓冲区
from rl.policies.actor import Gaussian_FF_Actor, Gaussian_LSTM_Actor  # 策略网络
from rl.policies.critic import FF_V, LSTM_V  # 价值网络
from rl.envs.normalize import get_normalization_params  # 状态归一化参数计算


class PPO:
    """
    PPO（Proximal Policy Optimization）算法实现类
    采用clip目标函数，支持并行采样、状态归一化、镜像对称损失、模仿学习等功能
    """
    def __init__(self, env_fn, args):
        """
        初始化PPO算法参数和网络
        
        参数:
            env_fn: 环境创建函数（用于获取环境信息和创建环境实例）
            args: 命令行参数，包含训练配置（如折扣因子、学习率、迭代次数等）
        """
        # 强化学习核心超参数
        self.gamma = args.gamma  # 折扣因子（未来奖励衰减系数）
        self.lam = args.lam  # GAE（广义优势估计）的平滑系数
        self.lr = args.lr  # 学习率
        self.eps = args.eps  # 优化器epsilon（数值稳定性参数）
        self.ent_coeff = args.entropy_coeff  # 熵惩罚系数（鼓励探索）
        self.clip = args.clip  # PPO剪辑系数（通常为0.2）
        self.minibatch_size = args.minibatch_size  # 每次更新的小批量大小
        self.epochs = args.epochs  # 每个迭代中用同一批数据更新的轮数
        self.max_traj_len = args.max_traj_len  # 单条轨迹的最大长度（防止过长）
        self.use_gae = args.use_gae  # 是否使用GAE计算优势函数
        self.n_proc = args.num_procs  # 并行采样的进程数
        self.grad_clip = args.max_grad_norm  # 梯度裁剪阈值（防止梯度爆炸）
        self.mirror_coeff = args.mirror_coeff  # 镜像对称损失的系数
        self.eval_freq = args.eval_freq  # 评估策略的频率（每多少迭代评估一次）
        self.recurrent = args.recurrent  # 是否使用循环网络（LSTM）
        self.imitate_coeff = args.imitate_coeff  # 模仿损失的系数

        # 总批量大小 = 并行进程数 * 单进程最大轨迹长度
        self.batch_size = self.n_proc * self.max_traj_len

        # 训练状态跟踪
        self.total_steps = 0  # 总训练步数
        self.highest_reward = -np.inf  # 记录最高评估奖励（用于保存最优模型）
        self.iteration_count = 0  # 训练迭代计数器

        # 日志和模型保存路径
        self.save_path = Path(args.logdir)
        Path.mkdir(self.save_path, parents=True, exist_ok=True)  # 创建目录（父目录不存在则创建）

        # 初始化TensorBoard日志写入器
        self.writer = SummaryWriter(log_dir=self.save_path, flush_secs=10)

        # 获取环境状态和动作维度
        obs_dim = env_fn().observation_space.shape[0]  # 状态维度
        action_dim = env_fn().action_space.shape[0]  # 动作维度

        # 加载预训练模型或新建模型
        if args.continued:
            # 从指定路径加载预训练的Actor和Critic
            path_to_actor = args.continued
            path_to_critic = Path(args.continued.parent, "critic" + str(args.continued).split('actor')[1])
            policy = torch.load(path_to_actor, weights_only=False)  # 加载策略网络
            critic = torch.load(path_to_critic, weights_only=False)  # 加载价值网络

            # 重置策略的动作噪声参数（不使用预训练的噪声参数）
            if args.learn_std:
                policy.stds = torch.nn.Parameter(args.std_dev * torch.ones(action_dim))
            else:
                policy.stds = args.std_dev * torch.ones(action_dim)
            print(f"已加载预训练Actor: {path_to_actor}")
            print(f"已加载预训练Critic: {path_to_critic}")
        else:
            # 新建策略网络和价值网络（根据是否循环网络选择类型）
            if args.recurrent:
                # 循环策略网络（LSTM）
                policy = Gaussian_LSTM_Actor(obs_dim, action_dim, init_std=args.std_dev, learn_std=args.learn_std)
                # 循环价值网络（LSTM）
                critic = LSTM_V(obs_dim)
            else:
                # 前馈策略网络（MLP）
                policy = Gaussian_FF_Actor(obs_dim, action_dim, init_std=args.std_dev, learn_std=args.learn_std, bounded=False)
                # 前馈价值网络（MLP）
                critic = FF_V(obs_dim)

            # 计算或加载状态归一化参数（均值和标准差）
            if hasattr(env_fn(), 'obs_mean') and hasattr(env_fn(), 'obs_std'):
                # 若环境已内置归一化参数，直接使用
                obs_mean, obs_std = env_fn().obs_mean, env_fn().obs_std
            else:
                # 否则通过随机采样计算状态的均值和标准差
                obs_mean, obs_std = get_normalization_params(
                    iter=args.input_norm_steps,
                    noise_std=1,
                    policy=policy,
                    env_fn=env_fn,
                    procs=args.num_procs
                )
            # 将归一化参数赋值给策略和价值网络（禁用梯度计算）
            with torch.no_grad():
                policy.obs_mean, policy.obs_std = map(torch.Tensor, (obs_mean, obs_std))
                critic.obs_mean = policy.obs_mean
                critic.obs_std = policy.obs_std

        # 初始化基础策略（用于模仿学习，若启用）
        self.base_policy = None
        if args.imitate:
            self.base_policy = torch.load(args.imitate, weights_only=False)  # 加载模仿的基准策略

        # 初始化旧策略（用于PPO的比率计算）
        self.old_policy = deepcopy(policy)
        self.policy = policy  # 当前策略（待更新）
        self.critic = critic  # 价值网络

    @staticmethod
    def save(nets, save_path, suffix=""):
        """
        保存网络参数
        
        参数:
            nets: 字典，键为网络名称（如"actor"），值为网络实例
            save_path: 保存路径
            suffix: 文件名后缀（用于区分不同迭代的模型）
        """
        filetype = ".pt"  # 文件后缀
        for name, net in nets.items():
            path = Path(save_path, name + suffix + filetype)
            torch.save(net, path)
            print(f"已保存{name}至: {path}")
        return

    @ray.remote  # 标记为ray远程函数，支持并行执行
    @torch.no_grad()  # 禁用梯度计算（采样阶段无需更新参数）
    @staticmethod
    def sample(env_fn, policy, critic, gamma, lam, iteration_count, max_steps, max_traj_len, deterministic):
        """
        采样轨迹数据（单进程）
        
        参数:
            env_fn: 环境创建函数
            policy: 策略网络
            critic: 价值网络
            gamma: 折扣因子
            lam: GAE系数
            iteration_count: 当前迭代次数
            max_steps: 最大采样步数
            max_traj_len: 单条轨迹的最大长度
            deterministic: 是否使用确定性策略（采样时为False，评估时为True）
        返回:
            采样的轨迹数据（通过PPOBuffer的get_data()获取）
        """
        env = env_fn()  # 创建环境实例
        env.robot.iteration_count = iteration_count  # 记录当前迭代次数（用于课程学习）

        # 初始化轨迹缓冲区（大小为max_traj_len的2倍，防止溢出）
        memory = PPOBuffer(policy.state_dim, policy.action_dim, gamma, lam, size=max_traj_len*2)
        memory_full = False  # 标记缓冲区是否已满

        while not memory_full:
            # 重置环境，获取初始状态
            state = torch.tensor(env.reset(), dtype=torch.float)
            done = False  # 轨迹终止标志
            traj_len = 0  # 当前轨迹长度

            # 初始化循环网络的隐藏状态（若使用LSTM）
            if hasattr(policy, 'init_hidden_state'):
                policy.init_hidden_state()
            if hasattr(critic, 'init_hidden_state'):
                critic.init_hidden_state()

            # 循环执行动作，直到轨迹终止或达到最大长度
            while not done and traj_len < max_traj_len:
                # 策略生成动作（deterministic控制是否随机）
                action = policy(state, deterministic=deterministic)
                # 价值网络估计状态价值
                value = critic(state)

                # 执行动作，获取下一状态、奖励、终止标志
                next_state, reward, done, _ = env.step(action.numpy().copy())

                # 将单步数据存入缓冲区
                reward = torch.tensor(reward, dtype=torch.float)
                memory.store(state, action, reward, value, done)
                # 检查缓冲区是否已满
                memory_full = (len(memory) >= max_steps)

                # 更新状态和轨迹长度
                state = torch.tensor(next_state, dtype=torch.float)
                traj_len += 1

            # 轨迹结束后，用最后一个状态的价值完成回报计算
            value = critic(state)
            memory.finish_path(last_val=(not done) * value)  # 若未终止，用价值估计作为后续回报

        # 返回收集的轨迹数据
        return memory.get_data()

    def sample_parallel(self, *args, deterministic=False):
        """
        并行采样轨迹数据（聚合多个进程的结果）
        
        参数:
            *args: 传递给sample方法的参数（env_fn, policy, critic等）
            deterministic: 是否使用确定性策略
        返回:
            聚合后的轨迹数据（封装为Data对象，属性为各类数据）
        """
        # 每个进程的最大采样步数 = 总批量 / 进程数
        max_steps = (self.batch_size // self.n_proc)
        # 组装传给sample方法的参数
        worker_args = (self.gamma, self.lam, self.iteration_count, max_steps, self.max_traj_len, deterministic)
        args = args + worker_args

        # 启动多个并行采样进程
        worker = self.sample
        workers = [worker.remote(*args) for _ in range(self.n_proc)]
        result = ray.get(workers)  # 获取所有进程的采样结果

        # 聚合所有进程的数据（按键拼接张量）
        keys = result[0].keys()
        aggregated_data = {
            k: torch.cat([r[k] for r in result]) for k in keys
        }

        # 将聚合数据封装为对象（方便通过属性访问）
        class Data:
            def __init__(self, data):
                for key, value in data.items():
                    setattr(self, key, value)
        data = Data(aggregated_data)
        return data

    def update_actor_critic(self, obs_batch, action_batch, return_batch, advantage_batch, mask, mirror_observation=None, mirror_action=None):
        """
        更新Actor（策略网络）和Critic（价值网络）
        
        参数:
            obs_batch: 状态批次（形状随网络类型变化）
            action_batch: 动作批次
            return_batch: 回报批次（GAE目标）
            advantage_batch: 优势函数批次
            mask: 掩码（用于处理变长序列，忽略填充部分）
            mirror_observation: 观测镜像函数（用于镜像对称损失）
            mirror_action: 动作镜像函数（用于镜像对称损失）
        返回:
            各类损失和指标（用于日志记录）
        """
        # 当前策略的动作分布及对数概率
        pdf = self.policy.distribution(obs_batch)
        log_probs = pdf.log_prob(action_batch).sum(-1, keepdim=True)  # 按动作维度求和

        # 旧策略的动作分布及对数概率（用于计算比率）
        old_pdf = self.old_policy.distribution(obs_batch)
        old_log_probs = old_pdf.log_prob(action_batch).sum(-1, keepdim=True)

        # 策略比率：新策略概率 / 旧策略概率（指数形式避免数值下溢）
        ratio = (log_probs - old_log_probs).exp()

        # PPO剪辑损失（clipped surrogate loss）
        cpi_loss = ratio * advantage_batch * mask  # 未剪辑的替代损失
        clip_loss = ratio.clamp(1.0 - self.clip, 1.0 + self.clip) * advantage_batch * mask  # 剪辑后的替代损失
        actor_loss = -torch.min(cpi_loss, clip_loss).mean()  # 取最小值的平均作为Actor损失

        # 剪辑比例（记录有多少样本被剪辑，用于监控）
        clip_fraction = torch.mean((torch.abs(ratio - 1) > self.clip).float()).item()

        # Critic损失（均方误差，用GAE目标作为标签）
        values = self.critic(obs_batch)
        critic_loss = F.mse_loss(return_batch, values)

        # 熵惩罚（鼓励策略探索，降低确定性）
        entropy_penalty = -(pdf.entropy() * mask).mean()

        # 镜像对称损失（利用环境对称性，提升策略鲁棒性）
        deterministic_actions = self.policy(obs_batch)  # 确定性动作（均值）
        if mirror_observation is not None and mirror_action is not None:
            if self.recurrent:
                # 循环网络需按时间步处理镜像观测
                mir_obs = torch.stack([mirror_observation(obs_batch[i,:,:]) for i in range(obs_batch.shape[0])])
                mirror_actions = self.policy(mir_obs)
            else:
                # 前馈网络直接处理批量镜像观测
                mir_obs = mirror_observation(obs_batch)
                mirror_actions = self.policy(mir_obs)
            mirror_actions = mirror_action(mirror_actions)  # 动作镜像变换
            mirror_loss = (deterministic_actions - mirror_actions).pow(2).mean()  # 均方误差
        else:
            mirror_loss = torch.zeros_like(actor_loss)  # 不启用则损失为0

        # 模仿损失（使当前策略接近基准策略，用于迁移学习）
        if self.base_policy is not None:
            imitation_loss = (self.base_policy(obs_batch) - deterministic_actions).pow(2).mean()
        else:
            imitation_loss = torch.zeros_like(actor_loss)  # 不启用则损失为0

        # 近似KL散度（用于早期停止，检测策略更新过大）
        with torch.no_grad():
            log_ratio = log_probs - old_log_probs
            approx_kl_div = torch.mean((ratio - 1) - log_ratio)  # 近似KL散度

        # 更新Actor网络
        self.actor_optimizer.zero_grad()  # 清零梯度
        # 总损失 = Actor损失 + 镜像损失*系数 + 模仿损失*系数 + 熵惩罚*系数
        (actor_loss + self.mirror_coeff*mirror_loss + self.imitate_coeff*imitation_loss + self.ent_coeff*entropy_penalty).backward(retain_graph=True)
        # 梯度裁剪（防止梯度爆炸）
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.grad_clip)
        self.actor_optimizer.step()  # 执行优化

        # 更新Critic网络
        self.critic_optimizer.zero_grad()  # 清零梯度
        critic_loss.backward(retain_graph=True)  # 计算梯度
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.grad_clip)
        self.critic_optimizer.step()  # 执行优化

        # 返回各类损失和指标
        return (
            actor_loss,
            entropy_penalty,
            critic_loss,
            approx_kl_div,
            mirror_loss,
            imitation_loss,
            clip_fraction,
        )

    def evaluate(self, env_fn, nets, itr, num_batches=5):
        """
        评估当前策略性能
        
        参数:
            env_fn: 环境创建函数
            nets: 包含Actor和Critic的字典
            itr: 当前迭代次数
            num_batches: 评估的批次数量
        返回:
            评估数据批次
        """
        # 将网络设为评估模式（禁用dropout等）
        for net in nets.values():
            net.eval()

        # 收集多个批次的评估数据
        eval_batches = []
        for _ in range(num_batches):
            # 确定性采样（不添加探索噪声）
            batch = self.sample_parallel(env_fn, *nets.values(), deterministic=True)
            eval_batches.append(batch)

        # 保存当前迭代的网络
        self.save(nets, self.save_path, "_" + repr(itr))

        # 若当前评估奖励为历史最高，保存为最优模型（actor.pt, critic.pt）
        eval_ep_rewards = [float(i) for batch in eval_batches for i in batch.ep_rewards]
        avg_eval_ep_rewards = np.mean(eval_ep_rewards)
        if self.highest_reward < avg_eval_ep_rewards:
            self.highest_reward = avg_eval_ep_rewards
            self.save(nets, self.save_path)

        return eval_batches

    def train(self, env_fn, n_itr):
        """
        主训练循环
        
        参数:
            env_fn: 环境创建函数
            n_itr: 总训练迭代次数
        """
        # 初始化优化器（Adam优化器）
        self.actor_optimizer = optim.Adam(self.policy.parameters(), lr=self.lr, eps=self.eps)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.lr, eps=self.eps)

        train_start_time = time.time()  # 记录训练开始时间

        # 获取环境的镜像函数（若有）
        obs_mirr, act_mirr = None, None
        if hasattr(env_fn(), 'mirror_observation'):
            obs_mirr = env_fn().mirror_clock_observation  # 带有时钟处理的观测镜像函数
        if hasattr(env_fn(), 'mirror_action'):
            act_mirr = env_fn().mirror_action  # 动作镜像函数

        # 主训练循环
        for itr in range(n_itr):
            print(f"********** 迭代 {itr} ************")

            # 将网络设为训练模式
            self.policy.train()
            self.critic.train()

            # 更新迭代计数器（用于课程学习）
            self.iteration_count = itr

            # 并行采样数据
            sample_start_time = time.time()
            policy_ref = ray.put(self.policy)  # 将策略网络存入ray共享内存
            critic_ref = ray.put(self.critic)  # 将价值网络存入ray共享内存
            batch = self.sample_parallel(env_fn, policy_ref, critic_ref)  # 采样并聚合数据
            # 提取采样数据（转换为float类型）
            observations = batch.states.float()
            actions = batch.actions.float()
            returns = batch.returns.float()
            values = batch.values.float()

            # 打印采样信息
            num_samples = len(observations)
            elapsed = time.time() - sample_start_time
            print(f"采样完成，耗时 {elapsed:.2f}s，共 {num_samples} 步。")

            # 计算优势函数并归一化（减去均值，除以标准差）
            advantages = returns - values  # 优势 = 回报 - 价值估计
            advantages = (advantages - advantages.mean()) / (advantages.std() + self.eps)  # 归一化

            # 计算小批量大小（默认等于总样本数）
            minibatch_size = self.minibatch_size or num_samples
            self.total_steps += num_samples  # 更新总步数

            # 将当前策略参数复制到旧策略（用于下一次比率计算）
            self.old_policy.load_state_dict(self.policy.state_dict())

            # 优化网络（多轮更新）
            optimizer_start_time = time.time()

            # 记录各类损失和指标
            actor_losses = []
            entropies = []
            critic_losses = []
            kls = []
            mirror_losses = []
            imitation_losses = []
            clip_fractions = []

            # 多轮更新（用同一批数据更新epochs次）
            for epoch in range(self.epochs):
                # 根据网络类型选择采样器（循环网络按轨迹采样，前馈网络按随机样本采样）
                if self.recurrent:
                    # 循环网络：按轨迹索引采样（保持时序连续性）
                    random_indices = SubsetRandomSampler(range(len(batch.traj_idx)-1))
                    sampler = BatchSampler(random_indices, minibatch_size, drop_last=False)
                else:
                    # 前馈网络：随机采样样本（无需保持时序）
                    random_indices = SubsetRandomSampler(range(num_samples))
                    sampler = BatchSampler(random_indices, minibatch_size, drop_last=True)

                # 遍历小批量
                for indices in sampler:
                    if self.recurrent:
                        # 循环网络：按轨迹索引提取变长序列，并用0填充至相同长度
                        obs_batch = [observations[int(batch.traj_idx[i]):int(batch.traj_idx[i+1])] for i in indices]
                        action_batch = [actions[int(batch.traj_idx[i]):int(batch.traj_idx[i+1])] for i in indices]
                        return_batch = [returns[int(batch.traj_idx[i]):int(batch.traj_idx[i+1])] for i in indices]
                        advantage_batch = [advantages[int(batch.traj_idx[i]):int(batch.traj_idx[i+1])] for i in indices]
                        mask = [torch.ones_like(r) for r in return_batch]  # 掩码（1表示有效，0表示填充）

                        # 填充序列（batch_first=False表示时序维度在前）
                        obs_batch = pad_sequence(obs_batch, batch_first=False)
                        action_batch = pad_sequence(action_batch, batch_first=False)
                        return_batch = pad_sequence(return_batch, batch_first=False)
                        advantage_batch = pad_sequence(advantage_batch, batch_first=False)
                        mask = pad_sequence(mask, batch_first=False)
                    else:
                        # 前馈网络：直接按索引提取批量数据
                        obs_batch = observations[indices]
                        action_batch = actions[indices]
                        return_batch = returns[indices]
                        advantage_batch = advantages[indices]
                        mask = 1  # 无需掩码（无填充）

                    # 更新网络并获取损失
                    scalars = self.update_actor_critic(
                        obs_batch, action_batch, return_batch, advantage_batch, mask,
                        mirror_observation=obs_mirr, mirror_action=act_mirr
                    )
                    # 解包损失和指标
                    actor_loss, entropy_penalty, critic_loss, approx_kl_div, mirror_loss, imitation_loss, clip_fraction = scalars

                    # 记录损失
                    actor_losses.append(actor_loss.item())
                    entropies.append(entropy_penalty.item())
                    critic_losses.append(critic_loss.item())
                    kls.append(approx_kl_div.item())
                    mirror_losses.append(mirror_loss.item())
                    imitation_losses.append(imitation_loss.item())
                    clip_fractions.append(clip_fraction)

            # 打印优化耗时
            elapsed = time.time() - optimizer_start_time
            print(f"优化完成，耗时 {elapsed:.2f}s")

            # 获取当前策略的动作噪声标准差
            action_noise = self.policy.stds.data.tolist()

            # 打印训练指标（格式化输出）
            sys.stdout.write("-" * 37 + "\n")
            sys.stdout.write("| %15s | %15s |" % ('平均回合奖励', f"{torch.mean(batch.ep_rewards):8.5g}") + "\n")
            sys.stdout.write("| %15s | %15s |" % ('平均回合长度', f"{torch.mean(batch.ep_lens):8.5g}") + "\n")
            sys.stdout.write("| %15s | %15s |" % ('Actor损失', f"{np.mean(actor_losses):8.3g}") + "\n")
            sys.stdout.write("| %15s | %15s |" % ('Critic损失', f"{np.mean(critic_losses):8.3g}") + "\n")
            sys.stdout.write("| %15s | %15s |" % ('镜像损失', f"{np.mean(mirror_losses):8.3g}") + "\n")
            sys.stdout.write("| %15s | %15s |" % ('模仿损失', f"{np.mean(imitation_losses):8.3g}") + "\n")
            sys.stdout.write("| %15s | %15s |" % ('平均KL散度', f"{np.mean(kls):8.3g}") + "\n")
            sys.stdout.write("| %15s | %15s |" % ('平均熵', f"{np.mean(entropies):8.3g}") + "\n")
            sys.stdout.write("| %15s | %15s |" % ('剪辑比例', f"{np.mean(clip_fractions):8.3g}") + "\n")
            sys.stdout.write("| %15s | %15s |" % ('平均噪声标准差', f"{np.mean(action_noise):8.3g}") + "\n")
            sys.stdout.write("-" * 37 + "\n")
            sys.stdout.flush()

            # 打印训练时间统计
            elapsed = time.time() - train_start_time
            iter_avg = elapsed/(itr+1)  # 平均迭代时间
            ETA = round((n_itr - itr)*iter_avg)  # 预计剩余时间
            print(f"总耗时: {elapsed:.2f}s，总步数: {self.total_steps}（帧率={self.total_steps/elapsed:.2f}）。平均迭代时间={iter_avg:.2f}s，预计剩余时间={datetime.timedelta(seconds=ETA)}")

            # 定期评估策略（首次迭代或达到评估频率）
            if itr == 0 or (itr + 1) % self.eval_freq == 0:
                nets = {"actor": self.policy, "critic": self.critic}  # 待评估的网络

                # 执行评估
                evaluate_start = time.time()
                eval_batches = self.evaluate(env_fn, nets, itr)
                eval_time = time.time() - evaluate_start

                # 计算评估指标
                eval_ep_lens = [float(i) for b in eval_batches for i in b.ep_lens]
                eval_ep_rewards = [float(i) for b in eval_batches for i in b.ep_rewards]
                avg_eval_ep_lens = np.mean(eval_ep_lens)
                avg_eval_ep_rewards = np.mean(eval_ep_rewards)
                print("====评估回合====")
                print(f"（回合长度: {avg_eval_ep_lens:.3f}，奖励: {avg_eval_ep_rewards:.3f}，耗时: {eval_time:.2f}s）")

                # 记录评估日志到TensorBoard
                self.writer.add_scalar("Eval/mean_reward", avg_eval_ep_rewards, itr)
                self.writer.add_scalar("Eval/mean_episode_length", avg_eval_ep_lens, itr)

            # 记录训练日志到TensorBoard
            self.writer.add_scalar("Loss/actor", np.mean(actor_losses), itr)
            self.writer.add_scalar("Loss/critic", np.mean(critic_losses), itr)
            self.writer.add_scalar("Loss/mirror", np.mean(mirror_losses), itr)
            self.writer.add_scalar("Loss/imitation", np.mean(imitation_losses), itr)
            self.writer.add_scalar("Train/mean_reward", torch.mean(batch.ep_rewards), itr)
            self.writer.add_scalar("Train/mean_episode_length", torch.mean(batch.ep_lens), itr)
            self.writer.add_scalar("Train/mean_noise_std", np.mean(action_noise), itr)