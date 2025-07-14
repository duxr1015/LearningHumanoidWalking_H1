import numpy as np
import transforms3d as tf3  # 用于3D变换的库
from tasks import rewards  # 导入奖励函数模块

class WalkingTask(object):
    """双足机器人的动态稳定行走任务。"""

    def __init__(self,
                 client=None,
                 dt=0.025,
                 neutral_foot_orient=[],
                 root_body='pelvis',
                 lfoot_body='lfoot',
                 rfoot_body='rfoot',
                 waist_r_joint='waist_r',
                 waist_p_joint='waist_p',
    ):
        """
        初始化行走任务
        
        参数:
            client: 与仿真环境交互的客户端对象（用于获取机器人状态、施加控制等）
            dt: 控制时间步长，默认0.025秒
            neutral_foot_orient: 足部的中性姿态（默认空列表）
            root_body: 根身体部位名称，默认'pelvis'（骨盆）
            lfoot_body: 左脚身体部位名称，默认'lfoot'
            rfoot_body: 右脚身体部位名称，默认'rfoot'
            waist_r_joint: 腰部旋转关节名称，默认'waist_r'
            waist_p_joint: 腰部俯仰关节名称，默认'waist_p'
        """
        self._client = client  # 仿真客户端
        self._control_dt = dt  # 控制时间步长
        self._neutral_foot_orient = neutral_foot_orient  # 足部中性姿态

        # 获取机器人总质量
        self._mass = self._client.get_robot_mass()

        # 以下参数依赖于具体机器人，目前硬编码
        # 理想情况下，这些应作为__init__的参数传入
        self._goal_speed_ref = []  # 目标速度参考值
        self._goal_height_ref = []  # 目标高度参考值
        self._swing_duration = []  # 摆动阶段持续时间
        self._stance_duration = []  # 支撑阶段持续时间
        self._total_duration = []  # 一个完整周期的总持续时间

        # 保存身体部位名称
        self._root_body_name = root_body        # 根身体（如骨盆）
        self._lfoot_body_name = lfoot_body      # 左脚
        self._rfoot_body_name = rfoot_body      # 右脚

    def calc_reward(self, prev_torque, prev_action, action):
        """
        计算当前步骤的奖励值
        
        参数:
            prev_torque: 上一步的关节力矩
            prev_action: 上一步的动作
            action: 当前步骤的动作
        返回:
            包含各奖励项的字典
        """
        # 获取左右脚的线速度（取第一个元素，可能为x方向速度）
        self.l_foot_vel = self._client.get_lfoot_body_vel()[0]
        self.r_foot_vel = self._client.get_rfoot_body_vel()[0]
        
        # 获取左右脚的地面反作用力（GRF：Ground Reaction Force）
        self.l_foot_frc = self._client.get_lfoot_grf()
        self.r_foot_frc = self._client.get_rfoot_grf()
        
        # 从时钟变量中提取左右脚的力和速度相关信息
        r_frc = self.right_clock[0]
        l_frc = self.left_clock[0]
        r_vel = self.right_clock[1]
        l_vel = self.left_clock[1]
        
        # 计算各奖励项并加权求和
        reward = dict(
            # 脚部力时钟奖励（占比15%）
            foot_frc_score=0.150 * rewards._calc_foot_frc_clock_reward(self, l_frc, r_frc),
            # 脚部速度时钟奖励（占比15%）
            foot_vel_score=0.150 * rewards._calc_foot_vel_clock_reward(self, l_vel, r_vel),
            # 身体姿态奖励（左右脚和根身体的平均，占比5%）
            orient_cost=0.050 * (rewards._calc_body_orient_reward(self, self._lfoot_body_name) +
                                 rewards._calc_body_orient_reward(self, self._rfoot_body_name) +
                                 rewards._calc_body_orient_reward(self, self._root_body_name)) / 3,
            # 根身体加速度奖励（占比5%）
            root_accel=0.050 * rewards._calc_root_accel_reward(self),
            # 高度误差奖励（占比5%）
            height_error=0.050 * rewards._calc_height_reward(self),
            # 质心前进速度误差奖励（占比20%）
            com_vel_error=0.200 * rewards._calc_fwd_vel_reward(self),
            # 力矩惩罚（占比5%）
            torque_penalty=0.050 * rewards._calc_torque_reward(self, prev_torque),
            # 动作惩罚（占比5%）
            action_penalty=0.050 * rewards._calc_action_reward(self, action, prev_action),
        )
        return reward

    def step(self):
        """
        推进任务的一个时间步（更新相位）
        """
        # 如果当前相位超过周期，则重置相位为0
        if self._phase > self._period:
            self._phase = 0
        # 相位递增（进入下一个控制步）
        self._phase += 1
        return

    def done(self):
        """
        判断当前episode是否结束（终止条件）
        
        返回:
            布尔值，True表示episode结束，False表示继续
        """
        # 检查机器人是否发生自碰撞
        contact_flag = self._client.check_self_collisions()
        # 获取机器人的关节位置
        qpos = self._client.get_qpos()
        
        # 定义终止条件：
        # - 根身体z坐标（高度）低于0.6米
        # - 根身体z坐标（高度）高于1.4米
        # - 发生自碰撞
        terminate_conditions = {
            "qpos[2]_ll": (qpos[2] < 0.6),  # qpos[2]通常为根身体的z坐标
            "qpos[2]_ul": (qpos[2] > 1.4),
            "contact_flag": contact_flag,
        }

        # 如果任何一个终止条件满足，则返回True（episode结束）
        done = True in terminate_conditions.values()
        return done

    def reset(self, iter_count=0):
        """
        重置任务状态（用于episode开始时）
        
        参数:
            iter_count: 迭代计数（默认0）
        """
        # 随机选择目标速度：0或0.3-0.4之间的随机值
        self._goal_speed_ref = np.random.choice([0, np.random.uniform(0.3, 0.4)])
        
        # 创建左右脚的相位奖励时钟（基于摆动和支撑阶段持续时间）
        self.right_clock, self.left_clock = rewards.create_phase_reward(
            self._swing_duration,
            self._stance_duration,
            0.1,
            "grounded",
            1 / self._control_dt  # 控制频率（1/时间步长）
        )

        # 计算一个完整周期的控制步数（包含左右脚各一次摆动）
        # 一个完整周期 = 左摆 + 右摆
        self._period = np.floor(2 * self._total_duration * (1 / self._control_dt))
        # 初始化时随机设置相位（增加训练多样性）
        self._phase = np.random.randint(0, self._period)