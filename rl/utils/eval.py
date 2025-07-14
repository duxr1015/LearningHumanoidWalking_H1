import torch
import time
from pathlib import Path  # 用于路径处理

import mujoco  # MuJoCo物理引擎库
import mujoco.viewer  # MuJoCo可视化工具

import imageio  # 用于视频写入
from datetime import datetime  # 用于生成时间戳


class EvaluateEnv:
    """
    评估环境类，用于加载训练好的策略并在环境中运行，同时录制视频
    """
    def __init__(self, env, policy, args):
        """
        参数:
            env: 待评估的环境实例（如H1机器人环境）
            policy: 训练好的策略网络（如Gaussian_FF_Actor）
            args: 命令行参数，包含评估配置（如 episode 长度、输出目录等）
        """
        self.env = env  # 环境实例
        self.policy = policy  # 策略网络
        self.ep_len = args.ep_len  # 评估的总时长（单位：秒）

        # 配置视频输出目录
        if args.out_dir is None:
            # 若未指定输出目录，默认保存到策略文件所在目录的"videos"子文件夹
            args.out_dir = Path(args.path.parent, "videos")

        video_outdir = Path(args.out_dir)  # 视频输出路径
        try:
            # 创建输出目录（若已存在则不报错）
            Path.mkdir(video_outdir, exist_ok=True)
            # 生成带时间戳的视频文件名（格式：策略文件名-时间戳.mp4）
            now = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
            video_fn = Path(video_outdir, args.path.stem + "-" + now + ".mp4")
            # 创建视频写入器（帧率60fps）
            self.writer = imageio.get_writer(video_fn, fps=60)
        except Exception as e:
            # 若视频创建失败，打印错误并退出
            print("无法创建视频写入器:", e)
            exit(-1)

    @torch.no_grad()  # 禁用梯度计算（评估阶段无需更新参数）
    def run(self):
        """运行评估流程：加载策略，在环境中执行动作，录制视频并保存"""
        # 配置渲染参数
        height = 480  # 渲染图像高度
        width = 640   # 渲染图像宽度
        # 创建MuJoCo渲染器（用于生成图像帧）
        renderer = mujoco.Renderer(self.env.model, height, width)
        # 启动被动式可视化窗口（仅显示，不接收用户交互）
        viewer = mujoco.viewer.launch_passive(self.env.model, self.env.data)
        frames = []  # 存储渲染的图像帧

        # 配置相机参数
        cam = viewer.cam  # 获取相机对象
        mujoco.mjv_defaultCamera(cam)  # 重置相机为默认参数
        cam.elevation = -20  # 相机仰角（负值表示俯视）
        cam.distance = 4     # 相机与目标的距离

        reset_counter = 0  # 记录重置环境的次数
        observation = self.env.reset()  # 重置环境，获取初始观测

        # 循环执行，直到达到指定的评估时长
        while self.env.data.time < self.ep_len:
            step_start = time.time()  # 记录当前步开始时间

            # 策略前向传播：输入观测，输出确定性动作（deterministic=True）
            raw = self.policy.forward(
                torch.tensor(observation, dtype=torch.float32), 
                deterministic=True
            ).detach().numpy()  # 转换为numpy数组（脱离计算图）

            # 在环境中执行动作，获取新观测、奖励、终止标志等
            observation, reward, done, _ = self.env.step(raw.copy())

            # 渲染场景并保存帧
            # 相机聚焦于机器人根节点（body(1)通常为躯干）
            cam.lookat = self.env.data.body(1).xpos.copy()
            # 更新渲染场景（使用当前相机参数）
            renderer.update_scene(self.env.data, cam)
            pixels = renderer.render()  # 生成图像帧
            frames.append(pixels)  # 保存帧

            viewer.sync()  # 同步可视化窗口（更新显示）

            # 若环境终止且重置次数小于3次，重置环境继续评估
            if done and reset_counter < 3:
                observation = self.env.reset()
                reset_counter += 1

            # 控制仿真步频与实时一致
            # 计算当前步应休眠的时间（确保仿真时间与现实时间同步）
            time_until_next_step = max(
                0,  # 确保休眠时间非负
                # 单个控制步包含的物理仿真步长总和（frame_skip * 物理时间步）
                self.env.frame_skip * self.env.model.opt.timestep 
                - (time.time() - step_start)  # 减去当前步已消耗的时间
            )
            time.sleep(time_until_next_step)  # 休眠以维持实时性

        # 评估结束后，将所有帧写入视频文件
        for frame in frames:
            self.writer.append_data(frame)
        self.writer.close()  # 关闭视频写入器
        self.env.close()     # 关闭环境
        viewer.close()       # 关闭可视化窗口
