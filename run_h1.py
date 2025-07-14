from pathlib import Path
import sys
import argparse
import ray
from functools import partial

import numpy as np
import torch
import torch.nn as nn
import pickle
import shutil

from rl.algos.ppo import PPO
from rl.envs.wrappers import SymmetricEnv
from rl.utils.eval import EvaluateEnv

def import_env(env_name_str):
    
    from envs.h1 import H1Env as Env
    
    return Env

def run_experiment(args):
    #------------ 环境初始化与包装 ------------

    # 导入环境类
    Env = import_env(args.env)
    # 创建环境函数(固定yaml配置路径)
    env_fn = partial(Env, path_to_yaml=args.yaml)
    _env = env_fn() # 临时实例化环境，用于后续数据增强

    # 尝试用SymmetricEnv包装环境(数据增强)
    if not args.no_mirror:
        try:
            env_fn = partial(SymmetricEnv, env_fn,
                             mirrored_obs =_env.robot.mirrored_obs, #镜像观测索引
                             mirrored_act=_env.robot.mirrored_acts, #镜像动作索引
                             clock_inds = _env.robot.clock_inds,) #时钟索引
        except Exception as e:
            print(f"Failed to wrap environment with SymmetricEnv: {e}")
    
    #---------------- 并行计算配置 --------------
    # 设置Ray并行计算
    if not ray.is_initialized():
        ray.init(num_cpus = args.num_procs)  # 初始化Ray，指定CPU数量     

    #---------------- 实验参数保存 --------------
    # 创建日志目录
    Path.mkdir(args.logdir, parents=True, exist_ok=True)
    # 保存超参数(args)到pickle文件
    pkl_path = Path(args.logdir,"experiments.pkl")
    with open(pkl_path, 'wb') as f:
        pickle.dump(args, f)
    # 复制配置文件到日志目录(方便复现)
    if args.yaml:
        shutil.copyfile(args.yaml,Path(args.logdir,"config.yaml"))

    #---------------- PPO算法初始化与训练 --------------
    algo = PPO(env_fn, args) # 初始化PPO算法
    algo.train(env_fn, args.n_itr) # 开始训练，迭代args.n_itr次




if __name__ == "__main__":

    #-------------- 训练参数解析 --------------
    parser = argparse.ArgumentParser()

    #训练模式
    if sys.argv[1] == 'train':
        sys.argv.remove(sys.argv[1]) # 移除'train'关键字
        # 定义训练相关参数
        parser.add_argument("--env", required=True, type=str)
        parser.add_argument("--logdir", default=Path("/tmp/logs"), type=Path, help="Path to save weights and logs")
        parser.add_argument("--input-norm-steps", type=int, default=100000)
        parser.add_argument("--n-itr", type=int, default=20000, help="Number of iterations of the learning algorithm")
        # PPO超参数
        parser.add_argument("--lr", type=float, default=1e-4, help="Adam learning rate") # Xie
        parser.add_argument("--eps", type=float, default=1e-5, help="Adam epsilon (for numerical stability)")
        parser.add_argument("--lam", type=float, default=0.95, help="Generalized advantage estimate discount")
        parser.add_argument("--gamma", type=float, default=0.99, help="MDP discount")
        parser.add_argument("--std-dev", type=float, default=0.223, help="Action noise for exploration")
        parser.add_argument("--learn-std", action="store_true", help="Exploration noise will be learned")
        parser.add_argument("--entropy-coeff", type=float, default=0.0, help="Coefficient for entropy regularization")
        parser.add_argument("--clip", type=float, default=0.2, help="Clipping parameter for PPO surrogate loss")
        parser.add_argument("--minibatch-size", type=int, default=64, help="Batch size for PPO updates")
        parser.add_argument("--epochs", type=int, default=3, help="Number of optimization epochs per PPO update") #Xie
        parser.add_argument("--use-gae", type=bool, default=True,help="Whether or not to calculate returns using Generalized Advantage Estimation")
        parser.add_argument("--num-procs", type=int, default=12, help="Number of threads to train on")
        parser.add_argument("--max-grad-norm", type=float, default=0.05, help="Value to clip gradients at")
        parser.add_argument("--max-traj-len", type=int, default=400, help="Max episode horizon")
        parser.add_argument("--no-mirror", required=False, action="store_true", help="to use SymmetricEnv")
        parser.add_argument("--mirror-coeff", required=False, default=0.4, type=float, help="weight for mirror loss")
        parser.add_argument("--eval-freq", required=False, default=100, type=int, help="Frequency of performing evaluation")
        parser.add_argument("--continued", required=False, type=Path, help="path to pretrained weights")
        parser.add_argument("--recurrent", required=False, action="store_true", help="use LSTM instead of FF")
        parser.add_argument("--imitate", required=False, type=str, default=None, help="Policy to imitate")
        parser.add_argument("--imitate-coeff", required=False, type=float, default=0.3, help="Coefficient for imitation loss")
        parser.add_argument("--yaml", required=False, type=str, default=None, help="Path to config file passed to Env class")
        args = parser.parse_args()

        run_experiment(args)
    
    elif sys.argv[1] == 'eval':
        sys.argv.remove(sys.argv[1])  # 移除'eval'关键字
        # 定义评估相关参数
        parser.add_argument("--path", default=Path("/tmp/logs"), type=Path)  # 模型路径
        parser.add_argument("--out-dir", default=None, type=Path)  # 评估结果（如视频）保存目录
        parser.add_argument("--ep-len", type=int, default=10)  # 评估episode长度（秒）
        args = parser.parse_args()
    
        # 定位模型文件（actor.pt是策略网络，critic.pt是价值网络）
        if args.path.is_file() and args.path.suffix==".pt":
            path_to_actor = args.path
        elif args.path.is_dir():
            path_to_actor = Path(args.path, "actor.pt")
        path_to_critic = Path(path_to_actor.parent, "critic" + str(path_to_actor).split('actor')[1])
        path_to_pkl = Path(path_to_actor.parent, "experiment.pkl")
    
        # 加载训练时的参数和模型
        run_args = pickle.load(open(path_to_pkl, "rb"))  # 训练时的args
        policy = torch.load(path_to_actor, weights_only=False)  # 加载策略网络
        critic = torch.load(path_to_critic, weights_only=False)  # 加载价值网络
        policy.eval()  # 切换到评估模式（禁用dropout等）
        critic.eval()
    
        # 导入环境并初始化评估器
        Env = import_env(run_args.env)
        env = partial(Env, run_args.yaml)() if run_args.yaml else Env()
        e = EvaluateEnv(env, policy, args)  # 评估环境
        e.run()  # 运行评估





