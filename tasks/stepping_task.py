import numpy as np
import random
import transforms3d as tf3  # 用于3D变换（如欧拉角与四元数转换）
from tasks import rewards  # 导入奖励函数模块
from enum import Enum, auto  # 用于定义枚举类

class WalkModes(Enum):
    """行走模式枚举类，定义机器人的多种行走状态"""
    STANDING = auto()  # 站立
    CURVED = auto()    # 曲线行走
    FORWARD = auto()   # 前进
    BACKWARD = auto()  # 后退
    INPLACE = auto()   # 原地踏步
    LATERAL = auto()   # 横向行走

class SteppingTask(object):
    """双足机器人通过踩目标点实现移动的任务类"""

    def __init__(self,
                 client=None,
                 dt=0.025,
                 neutral_foot_orient=[],
                 root_body='pelvis',
                 lfoot_body='lfoot',
                 rfoot_body='rfoot',
                 head_body='head',
    ):
        """
        初始化步进任务
        
        参数:
            client: 与仿真环境交互的客户端（用于获取机器人状态、控制仿真等）
            dt: 控制时间步长，默认0.025秒
            neutral_foot_orient: 足部的中性姿态（默认空列表）
            root_body: 根身体部位名称，默认'pelvis'（骨盆）
            lfoot_body: 左脚身体部位名称，默认'lfoot'
            rfoot_body: 右脚身体部位名称，默认'rfoot'
            head_body: 头部身体部位名称，默认'head'
        """
        self._client = client  # 仿真客户端
        self._control_dt = dt  # 控制时间步长

        self._mass = self._client.get_robot_mass()  # 机器人总质量

        # 初始化运动参数
        self._goal_speed_ref = 0  # 目标速度参考值
        self._goal_height_ref = []  # 目标高度参考值
        self._swing_duration = []  # 摆动阶段持续时间
        self._stance_duration = []  # 支撑阶段持续时间
        self._total_duration = []  # 一个完整周期的总持续时间

        # 保存身体部位名称
        self._head_body_name = head_body        # 头部
        self._root_body_name = root_body        # 根身体（如骨盆）
        self._lfoot_body_name = lfoot_body      # 左脚
        self._rfoot_body_name = rfoot_body      # 右脚

        # 读取预先生成的脚步计划文件
        with open('utils/footstep_plans.txt', 'r') as fn:
            lines = [l.strip() for l in fn.readlines()]  # 读取并清洗每行数据
        self.plans = []  # 存储所有脚步计划
        sequence = []    # 临时存储单个脚步序列
        for line in lines:
            if line == '---':  # 分隔符，代表一个脚步计划结束
                if len(sequence):
                    self.plans.append(sequence)  # 保存完整序列
                sequence = []  # 重置临时序列
                continue
            else:
                # 将每行数据转换为数组（x,y,z,theta）并添加到序列
                sequence.append(np.array([float(l) for l in line.split(',')]))

    def step_reward(self):
        """
        计算步进奖励（鼓励机器人踩到目标点）
        
        返回:
            综合目标命中奖励和进度奖励的步进奖励
        """
        # 获取当前目标点位置（前3个元素为x,y,z）
        target_pos = self.sequence[self.t1][0:3]
        # 计算左右脚到目标点的最小距离
        foot_dist_to_target = min([np.linalg.norm(ft - target_pos) for ft in [self.l_foot_pos,
                                                                            self.r_foot_pos]])
        hit_reward = 0  # 目标命中奖励
        if self.target_reached:
            # 命中目标时，根据距离计算奖励（距离越近奖励越高）
            hit_reward = np.exp(-foot_dist_to_target / 0.25)

        # 计算目标中点（当前目标与下一个目标的中点）
        target_mp = (self.sequence[self.t1][0:2] + self.sequence[self.t2][0:2]) / 2
        # 获取根身体的x,y坐标
        root_xy_pos = self._client.get_object_xpos_by_name(self._root_body_name, 'OBJ_BODY')[0:2]
        # 计算根身体到目标中点的距离
        root_dist_to_target = np.linalg.norm(root_xy_pos - target_mp)
        # 进度奖励（越接近目标中点奖励越高）
        progress_reward = np.exp(-root_dist_to_target / 2)
        
        # 综合命中奖励（80%）和进度奖励（20%）
        return (0.8 * hit_reward + 0.2 * progress_reward)

    def calc_reward(self, prev_torque, prev_action, action):
        """
        计算当前步骤的总奖励（综合多种奖励项）
        
        参数:
            prev_torque: 上一步的关节力矩
            prev_action: 上一步的动作
            action: 当前步骤的动作
        返回:
            包含各奖励项的字典
        """
        # 将目标航向角（sequence中的theta）转换为四元数
        orient = tf3.euler.euler2quat(0, 0, self.sequence[self.t1][3])
        # 从时钟变量中提取左右脚的力和速度相关函数
        r_frc = self.right_clock[0]
        l_frc = self.left_clock[0]
        r_vel = self.right_clock[1]
        l_vel = self.left_clock[1]
        
        # 站立模式下，强制设置力和速度的时钟函数（鼓励双脚用力，禁止移动）
        if self.mode == WalkModes.STANDING:
            r_frc = (lambda _: 1)    # 右脚力奖励函数（恒为1）
            l_frc = (lambda _: 1)    # 左脚力奖励函数（恒为1）
            r_vel = (lambda _: -1)   # 右脚速度惩罚函数（恒为-1）
            l_vel = (lambda _: -1)   # 左脚速度惩罚函数（恒为-1）
        
        # 获取头部和根身体的x,y坐标
        head_pos = self._client.get_object_xpos_by_name(self._head_body_name, 'OBJ_BODY')[0:2]
        root_pos = self._client.get_object_xpos_by_name(self._root_body_name, 'OBJ_BODY')[0:2]
        
        # 综合各奖励项（带权重）
        reward = dict(
            # 脚部力时钟奖励（15%）
            foot_frc_score=0.150 * rewards._calc_foot_frc_clock_reward(self, l_frc, r_frc),
            # 脚部速度时钟奖励（15%）
            foot_vel_score=0.150 * rewards._calc_foot_vel_clock_reward(self, l_vel, r_vel),
            # 根身体姿态奖励（5%，基于目标航向）
            orient_cost=0.050 * rewards._calc_body_orient_reward(self,
                                                                 self._root_body_name,
                                                                 quat_ref=orient),
            # 高度奖励（5%）
            height_error=0.050 * rewards._calc_height_reward(self),
            # 步进奖励（45%，鼓励踩目标点）
            step_reward=0.450 * self.step_reward(),
            # 上半身奖励（5%，鼓励头部与根身体位置接近，保持直立）
            upper_body_reward=0.050 * np.exp(-10 * np.square(np.linalg.norm(head_pos - root_pos)))
        )
        return reward

    def transform_sequence(self, sequence):
        """
        将脚步序列从机器人本地坐标系转换为世界坐标系
        
        参数:
            sequence: 本地坐标系下的脚步序列（相对根身体）
        返回:
            世界坐标系下的脚步序列
        """
        # 获取左右脚当前位置
        lfoot_pos = self._client.get_lfoot_body_pos()
        rfoot_pos = self._client.get_rfoot_body_pos()
        # 获取根身体的偏航角（绕z轴旋转角）
        root_yaw = tf3.euler.quat2euler(self._client.get_object_xquat_by_name(self._root_body_name, 'OBJ_BODY'))[2]
        # 计算左右脚中点（作为本地坐标系原点）
        mid_pt = (lfoot_pos + rfoot_pos) / 2
        sequence_rel = []  # 存储转换后的序列
        for x, y, z, theta in sequence:
            # 应用旋转变换（将本地x,y转换到世界坐标系）
            x_ = mid_pt[0] + x * np.cos(root_yaw) - y * np.sin(root_yaw)
            y_ = mid_pt[1] + x * np.sin(root_yaw) + y * np.cos(root_yaw)
            theta_ = root_yaw + theta  # 航向角叠加根身体偏航角
            step = np.array([x_, y_, z, theta_])  # 世界坐标系下的步点
            sequence_rel.append(step)
        return sequence_rel

    def generate_step_sequence(self, **kwargs):
        """
        根据参数生成脚步序列（支持多种行走模式）
        
        参数:
           ** kwargs: 包含步长、步距、步高、步数等参数的字典
        返回:
            生成的本地坐标系下的脚步序列
        """
        # 解析参数（步长、步距、步高、步数、是否曲线、是否横向）
        step_size, step_gap, step_height, num_steps, curved, lateral = kwargs.values()
        
        if curved:
            # 曲线行走模式：从预定义计划中随机选择，高度设为0
            plan = random.choice(self.plans)
            sequence = [[s[0], s[1], 0, s[2]] for s in plan]
            return np.array(sequence)

        if lateral:
            # 横向行走模式：左右交替迈步
            sequence = []
            y = 0  # 横向位移
            c = np.random.choice([-1, 1])  # 随机选择左右方向
            for i in range(1, num_steps):
                if i % 2:  # 奇数步：y增加
                    y += step_size
                else:  # 偶数步：y减少（部分恢复）
                    y -= (2/3) * step_size
                # 生成横向步点（x=0，y为计算值）
                step = np.array([0, c * y, 0, 0])
                sequence.append(step)
            return sequence

        # 前进/后退/原地模式：生成前后交替的脚步序列
        sequence = []
        # 根据初始相位决定第一步方向（左或右）
        if self._phase == (0.5 * self._period):
            # 相位为周期一半时，第一步向左
            first_step = np.array([0, -1 * np.random.uniform(0.095, 0.105), 0, 0])
            y = -step_gap
        else:
            # 其他相位时，第一步向右
            first_step = np.array([0, 1 * np.random.uniform(0.095, 0.105), 0, 0])
            y = step_gap
        sequence.append(first_step)  # 添加第一步
        
        x, z = 0, 0  # x为前后位移，z为高度
        c = np.random.randint(2, 4)  # 随机决定前几步不增加高度
        for i in range(1, num_steps - 1):
            x += step_size  # 前后位移累加
            y *= -1  # 左右方向交替
            if i > c:  # 超过c步后开始增加高度
                z += step_height
            # 生成步点
            step = np.array([x, y, z, 0])
            sequence.append(step)
        # 添加最后一步（方向与上一步相反）
        final_step = np.array([x + step_size, -y, z, 0])
        sequence.append(final_step)
        return sequence

    def update_goal_steps(self):
        """更新目标步点在机器人本地坐标系中的位置和姿态"""
        # 初始化目标步点参数（x,y,z,theta各两个目标）
        self._goal_steps_x[:] = np.zeros(2)
        self._goal_steps_y[:] = np.zeros(2)
        self._goal_steps_z[:] = np.zeros(2)
        self._goal_steps_theta[:] = np.zeros(2)
        
        # 获取根身体的位置和姿态（世界坐标系）
        root_pos = self._client.get_object_xpos_by_name(self._root_body_name, 'OBJ_BODY')
        root_quat = self._client.get_object_xquat_by_name(self._root_body_name, 'OBJ_BODY')
        
        # 处理当前两个目标步点（t1和t2）
        for idx, t in enumerate([self.t1, self.t2]):
            # 构建根身体的变换矩阵（从本地到世界）
            ref_frame = tf3.affines.compose(root_pos, tf3.quaternions.quat2mat(root_quat), np.ones(3))
            # 目标步点在世界坐标系中的位置和姿态
            abs_goal_pos = self.sequence[t][0:3]
            abs_goal_rot = tf3.euler.euler2mat(0, 0, self.sequence[t][3])
            absolute_target = tf3.affines.compose(abs_goal_pos, abs_goal_rot, np.ones(3))
            # 将世界坐标系目标转换为本地坐标系（根身体视角）
            relative_target = np.linalg.inv(ref_frame).dot(absolute_target)
            
            # 非站立模式下更新目标参数
            if self.mode != WalkModes.STANDING:
                self._goal_steps_x[idx] = relative_target[0, 3]  # x坐标
                self._goal_steps_y[idx] = relative_target[1, 3]  # y坐标
                self._goal_steps_z[idx] = relative_target[2, 3]  # z坐标
                # 偏航角（绕z轴）
                self._goal_steps_theta[idx] = tf3.euler.mat2euler(relative_target[:3, :3])[2]
        return

    def update_target_steps(self):
        """更新当前目标步点索引（t1递进，t2跟进）"""
        assert len(self.sequence) > 0  # 确保序列非空
        self.t1 = self.t2  # 当前目标更新为下一个目标
        self.t2 += 1  # 下一个目标索引+1
        # 若t2超出序列长度，固定为最后一个步点
        if self.t2 == len(self.sequence):
            self.t2 = len(self.sequence) - 1
        return

    def step(self):
        """
        推进任务的一个时间步（更新相位、状态、目标检测）
        """
        # 更新相位（超出周期则重置）
        self._phase += 1
        if self._phase >= self._period:
            self._phase = 0

        # 获取左右脚的姿态（四元数）、位置、速度和地面反作用力
        self.l_foot_quat = self._client.get_object_xquat_by_name('lf_force', 'OBJ_SITE')
        self.r_foot_quat = self._client.get_object_xquat_by_name('rf_force', 'OBJ_SITE')
        self.l_foot_pos = self._client.get_object_xpos_by_name('lf_force', 'OBJ_SITE')
        self.r_foot_pos = self._client.get_object_xpos_by_name('rf_force', 'OBJ_SITE')
        self.l_foot_vel = self._client.get_lfoot_body_vel()[0]
        self.r_foot_vel = self._client.get_rfoot_body_vel()[0]
        self.l_foot_frc = self._client.get_lfoot_grf()
        self.r_foot_frc = self._client.get_rfoot_grf()

        # 检测是否到达当前目标步点
        target_pos = self.sequence[self.t1][0:3]
        # 计算左右脚到目标点的距离
        foot_dist_to_target = min([np.linalg.norm(ft - target_pos) for ft in [self.l_foot_pos,
                                                                            self.r_foot_pos]])

        # 判断左右脚是否进入目标区域（距离小于目标半径）
        lfoot_in_target = (np.linalg.norm(self.l_foot_pos - target_pos) < self.target_radius)
        rfoot_in_target = (np.linalg.norm(self.r_foot_pos - target_pos) < self.target_radius)
        
        if lfoot_in_target or rfoot_in_target:
            self.target_reached = True  # 标记目标已到达
            self.target_reached_frames += 1  # 累计到达帧数
        else:
            self.target_reached = False  # 未到达目标
            self.target_reached_frames = 0  # 重置累计帧数

        # 若目标持续到达一定帧数，更新目标步点
        if self.target_reached and (self.target_reached_frames >= self.delay_frames):
            self.update_target_steps()
            self.target_reached = False
            self.target_reached_frames = 0

        # 更新目标步点参数
        self.update_goal_steps()
        return

    def substep(self):
        """子步骤处理（预留接口，暂未实现）"""
        pass

    def done(self):
        """
        判断当前episode是否结束（终止条件）
        
        返回:
            布尔值，True表示episode结束，False表示继续
        """
        # 检查机器人是否发生自碰撞
        contact_flag = self._client.check_self_collisions()

        # 获取根身体位置和双脚最低高度
        qpos = self._client.get_object_xpos_by_name(self._root_body_name, 'OBJ_BODY')
        foot_pos = min([c[2] for c in (self.l_foot_pos, self.r_foot_pos)])
        # 计算根身体相对高度（根z坐标 - 双脚最低z坐标）
        root_rel_height = qpos[2] - foot_pos
        
        # 定义终止条件：
        # - 根身体相对高度低于0.6米（可能摔倒）
        # - 发生自碰撞
        terminate_conditions = {
            "qpos[2]_ll": (root_rel_height < 0.6),
            "contact_flag": contact_flag,
        }

        # 若任何终止条件满足，返回True（episode结束）
        done = True in terminate_conditions.values()
        return done

    def reset(self, iter_count=0):
        """
        重置任务状态（用于新episode开始）
        
        参数:
            iter_count: 训练迭代次数（默认0）
        """
        self.iteration_count = iter_count  # 记录迭代次数

        # 初始化目标步点参数（x,y,z,theta各两个目标）
        self._goal_steps_x = [0, 0]
        self._goal_steps_y = [0, 0]
        self._goal_steps_z = [0, 0]
        self._goal_steps_theta = [0, 0]

        self.target_radius = 0.20  # 目标点半径（进入此范围视为到达）
        # 延迟帧数（目标持续到达该帧数后才更新目标）
        self.delay_frames = int(np.floor(self._swing_duration / self._control_dt))
        self.target_reached = False  # 目标是否到达标记
        self.target_reached_frames = 0  # 目标到达持续帧数
        self.t1 = 0  # 当前目标步点索引
        self.t2 = 0  # 下一个目标步点索引

        # 创建左右脚的相位奖励时钟（基于摆动和支撑阶段）
        self.right_clock, self.left_clock = rewards.create_phase_reward(
            self._swing_duration,
            self._stance_duration,
            0.1,
            "grounded",
            1 / self._control_dt  # 控制频率（1/时间步长）
        )

        # 计算一个完整周期的控制步数（包含左右脚各一次摆动）
        self._period = np.floor(2 * self._total_duration * (1 / self._control_dt))
        # 初始化时随机选择相位（0或周期一半，增加训练多样性）
        self._phase = int(np.random.choice([0, self._period / 2]))

        ## 生成脚步序列
        # 随机选择行走模式（带概率）
        self.mode = np.random.choice(
            [WalkModes.CURVED, WalkModes.STANDING, WalkModes.BACKWARD, WalkModes.LATERAL, WalkModes.FORWARD],
            p=[0.15, 0.05, 0.2, 0.3, 0.3]  # 各模式概率
        )

        # 初始化脚步序列参数
        d = {'step_size': 0.3, 'step_gap': 0.15, 'step_height': 0, 'num_steps': 20, 'curved': False, 'lateral': False}
        # 根据模式调整参数
        if self.mode == WalkModes.CURVED:
            d['curved'] = True  # 曲线模式
        elif self.mode == WalkModes.STANDING:
            d['num_steps'] = 1  # 站立模式（仅1步）
        elif self.mode == WalkModes.BACKWARD:
            d['step_size'] = -0.1  # 后退模式（步长为负）
        elif self.mode == WalkModes.INPLACE:
            # 原地模式（步长随机小范围波动）
            ss = np.random.uniform(-0.05, 0.05)
            d['step_size'] = ss
        elif self.mode == WalkModes.LATERAL:
            d['step_size'] = 0.4  # 横向模式
            d['lateral'] = True
        elif self.mode == WalkModes.FORWARD:
            # 前进模式（步高随训练迭代增加，最大0.1）
            h = np.clip((self.iteration_count - 3000) / 8000, 0, 1) * 0.1
            d['step_height'] = np.random.choice([-h, h])  # 随机上下波动
        else:
            raise Exception("无效的行走模式")
        
        # 生成并转换脚步序列
        sequence = self.generate_step_sequence(** d)
        self.sequence = self.transform_sequence(sequence)
        self.update_target_steps()  # 更新初始目标

        ## 创建地形（使用几何模型）
        nboxes = 20  # 地形方块数量
        boxes = ["box" + repr(i + 1).zfill(2) for i in range(nboxes)]  # 方块名称列表
        # 初始化序列（前半部分用生成的脚步序列，后半部分设为无效值）
        sequence = [np.array([0, 0, -1, 0]) for i in range(nboxes)]
        sequence[:len(self.sequence)] = self.sequence
        # 逐个设置方块位置和姿态（作为目标点标记）
        for box, step in zip(boxes, sequence):
            box_h = self._client.model.geom(box).size[2]  # 方块高度
            # 设置方块位置（目标点z坐标减去方块高度，确保顶部与目标点齐平）
            self._client.model.body(box).pos[:] = step[0:3] - np.array([0, 0, box_h])
            # 设置方块姿态（与目标点航向一致）
            self._client.model.body(box).quat[:] = tf3.euler.euler2quat(0, 0, step[3])
            self._client.model.geom(box).size[:] = np.array([0.15, 1, box_h])  # 方块大小
            self._client.model.geom(box).rgba[:] = np.array([0.8, 0.8, 0.8, 1])  # 方块颜色（灰色）

        # 特殊处理：前进模式下隐藏地面（避免遮挡目标方块）
        self._client.model.body('floor').pos[:] = np.array([0, 0, 0])
        if self.mode == WalkModes.FORWARD:
            self._client.model.body('floor').pos[:] = np.array([0, 0, -2])  # 地面下移2米（不可见）