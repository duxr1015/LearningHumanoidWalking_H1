import numpy as np
import transforms3d as tf3  # 用于3D变换计算
from tasks import rewards  # 导入奖励函数模块

class WalkingTask(object):
    """双足机器人动态稳定行走任务类"""

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
            client: 与仿真环境交互的客户端
            dt: 控制时间步长(秒)
            neutral_foot_orient: 足部中性姿态
            root_body: 根身体部位名称(如骨盆)
            lfoot_body: 左脚部位名称
            rfoot_body: 右脚部位名称
            waist_r_joint: 腰部侧屈关节名称
            waist_p_joint: 腰部俯仰关节名称
        """
        self._client = client  # 仿真客户端
        self._control_dt = dt  # 控制时间步长
        self._neutral_foot_orient = neutral_foot_orient  # 足部中性姿态

        self._mass = self._client.get_robot_mass()  # 获取机器人质量

        # 这些参数依赖于机器人，目前硬编码
        # 理想情况下应作为__init__参数传入
        self._goal_speed_ref = []  # 目标速度参考
        self._goal_height_ref = []  # 目标高度参考
        self._swing_duration = []  # 摆动阶段持续时间
        self._stance_duration = []  # 支撑阶段持续时间
        self._total_duration = []  # 一个完整周期的总持续时间

        self._root_body_name = root_body  # 根身体名称
        self._lfoot_body_name = lfoot_body  # 左脚名称
        self._rfoot_body_name = rfoot_body  # 右脚名称

    def calc_reward(self, prev_torque, prev_action, action):
        """
        计算当前状态的奖励值
        
        参数:
            prev_torque: 上一时间步的关节力矩
            prev_action: 上一时间步的动作
            action: 当前时间步的动作
        返回:
            包含各奖励项的字典
        """
        # 获取左右脚的速度和地面反作用力
        self.l_foot_vel = self._client.get_lfoot_body_vel()[0]
        self.r_foot_vel = self._client.get_rfoot_body_vel()[0]
        self.l_foot_frc = self._client.get_lfoot_grf()
        self.r_foot_frc = self._client.get_rfoot_grf()
        
        # 获取左右脚的相位时钟函数(力和速度)
        r_frc = self.right_clock[0]
        l_frc = self.left_clock[0]
        r_vel = self.right_clock[1]
        l_vel = self.right_clock[1]
        
        # 计算综合奖励(包含多个加权项)
        reward = dict(
            # 脚部力分布奖励(15%)
            foot_frc_score=0.150 * rewards._calc_foot_frc_clock_reward(self, l_frc, r_frc),
            # 脚部速度控制奖励(15%)
            foot_vel_score=0.150 * rewards._calc_foot_vel_clock_reward(self, l_vel, r_vel),
            # 姿态定向奖励(5%): 平均左脚、右脚和根身体的姿态奖励
            orient_cost=0.050 * (rewards._calc_body_orient_reward(self, self._lfoot_body_name) +
                                rewards._calc_body_orient_reward(self, self._rfoot_body_name) +
                                rewards._calc_body_orient_reward(self, self._root_body_name)) / 3,
            # 根身体加速度惩罚(5%)
            root_accel=0.050 * rewards._calc_root_accel_reward(self),
            # 高度误差惩罚(5%)
            height_error=0.050 * rewards._calc_height_reward(self),
            # 质心速度误差惩罚(20%)
            com_vel_error=0.200 * rewards._calc_fwd_vel_reward(self),
            # 关节力矩惩罚(5%)
            torque_penalty=0.050 * rewards._calc_torque_reward(self, prev_torque),
            # 动作平滑度惩罚(5%)
            action_penalty=0.050 * rewards._calc_action_reward(self, action, prev_action),
        )
        return reward

    def step(self):
        """
        执行一个时间步的任务更新
        """
        # 更新相位: 如果超过周期则重置
        if self._phase > self._period:
            self._phase = 0
        self._phase += 1
        return

    def done(self):
        """
        判断当前episode是否结束
        
        返回:
            True: 任务结束, False: 任务继续
        """
        # 检查是否发生自碰撞
        contact_flag = self._client.check_self_collisions()
        # 获取关节位置
        qpos = self._client.get_qpos()
        
        # 定义终止条件
        terminate_conditions = {
            "qpos[2]_ll": (qpos[2] < 0.6),  # 根身体高度过低
            "qpos[2]_ul": (qpos[2] > 1.4),  # 根身体高度过高
            "contact_flag": contact_flag,    # 发生自碰撞
        }

        # 只要有一个条件满足则任务结束
        done = True in terminate_conditions.values()
        return done

    def reset(self, iter_count=0):
        """
        重置任务状态(开始新的episode)
        
        参数:
            iter_count: 当前训练迭代次数
        """
        # 随机选择目标速度(静止或向前移动)
        self._goal_speed_ref = np.random.choice([0, np.random.uniform(0.3, 0.4)])
        
        # 创建左右脚的相位奖励时钟函数
        self.right_clock, self.left_clock = rewards.create_phase_reward(
            self._swing_duration,  # 摆动阶段持续时间
            self._stance_duration,  # 支撑阶段持续时间
            0.1,  # 相位偏移
            "grounded",  # 模式: 接地
            1 / self._control_dt  # 频率(1/时间步长)
        )

        # 计算一个完整周期的控制步数(左右脚各一次摆动)
        self._period = np.floor(2 * self._total_duration * (1 / self._control_dt))
        # 初始化时随机化相位, 增加训练多样性
        self._phase = np.random.randint(0, self._period)