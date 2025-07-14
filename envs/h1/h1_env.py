import os 
import copy
import numpy as np
import transforms3d as np
import collections

from robots.robot_base import RobotBase
from envs.common import mujoco_env 
from envs.common import robot_interface
from envs.common import config_builder
import mujoco

from .gen_xml import *

class Task:
    def __init__(self, client, neutral_pose):
        self._client = client  # 机器人接口，用于获取状态
        self.neutral_pose = neutral_pose  # 中立姿态（参考姿势）

    def calc_reward(self,prev_torque,prev_action,action):
        ######获取机器人实时状态
        # 获取盆骨(躯干根节点)的位姿(仿射矩阵)
        root_pose = self._client.get_object_affine_by_name("pelvis",'OBJ_BODY')
        # 获取头部位置(用于上半身姿态判断)
        head_pose = self._client.get_object_affine_by_name("head",'OBJ_GEOM')
        # 获取当前关节位置（前10个关节，可能对应腿部）
        current_pose = np.array(self._client.get_act_joint_positions())[:10]
        # 获取关节力矩
        tau_error = np.linalg.norm(self._client.get_act_joint_torques())
        # 获取骨盆速度（局部坐标系，前2个角速度分量）
        root_vel = self._client.get_body_vel("pelvis", frame=1)[0][:2]
        # 获取偏航角速度（qvel[5]可能对应yaw轴速度）
        yaw_vel = self._client.get_qvel()[5]

        ######计算各误差项（与目标的差距）
        # 1. 高度误差：骨盆高度与目标高度（0.98m）的差
        target_root_h = 0.98
        root_h = root_pose[2, 3]  # 仿射矩阵第3行第4列是z坐标（高度）
        height_error = np.linalg.norm(root_h - target_root_h)
        # 2. 上半身误差：头部相对骨盆的偏移（希望头部保持在骨盆正上方）
        head_pose_offset = np.zeros(2)
        head_pos_in_robot_base = np.linalg.inv(root_pose).dot(head_pose)[:2, 3] - head_pose_offset
        upperbody_error = np.linalg.norm(head_pos_in_robot_base)  # x/y方向偏移的模长

        # 3. 姿势误差：当前关节位置与中立姿态的差（鼓励接近自然姿势）
        posture_error = np.linalg.norm(current_pose - self.neutral_pose)

        # 4. 力矩误差：关节力矩的模长（鼓励低力矩消耗，节省能量）
        tau_error = np.linalg.norm(self._client.get_act_joint_torques())

        # 5. 前进速度误差：骨盆局部坐标系下的前向速度（鼓励稳定前进）
        fwd_vel_error = np.linalg.norm(root_vel)  # 前向速度的模长

        # 6. 偏航速度误差：绕z轴的旋转速度（鼓励直线行走，减少转弯）
        yaw_vel_error = np.linalg.norm(yaw_vel)

        ###########误差转换为奖励（核心设计）
        reward = {
            # 前向速度奖励：权重0.3，误差越小奖励越高（鼓励稳定前进）
            "com_vel_error": 0.3 * np.exp(-4 * np.square(fwd_vel_error)),
            # 偏航速度奖励：权重0.3，鼓励低偏航（直线行走）
            "yaw_vel_error": 0.3 * np.exp(-4 * np.square(yaw_vel_error)),
            # 高度奖励：权重0.1，鼓励骨盆高度接近0.98m（稳定站立）
            "height": 0.1 * np.exp(-0.5 * np.square(height_error)),
            # 上半身奖励：权重0.1，强烈惩罚头部偏移（系数40，鼓励直立）
            "upperbody": 0.1 * np.exp(-40*np.square(upperbody_error)),
            # 力矩奖励：权重0.1，鼓励低力矩（系数5e-5，节省能量）
            "joint_torque_reward": 0.1 * np.exp(-5e-5*np.square(tau_error)),
            # 姿势奖励：权重0.1，鼓励关节接近中立姿态
            "posture": 0.1 * np.exp(-1*np.square(posture_error)),
        }

        return reward
    
    def step(self):
        pass      # 可扩展：每步任务逻辑（如生成目标速度、更新环境）

    def substep(self):
        pass      # 可扩展：子步骤逻辑（如高频控制信号调整）

    def done(self):
        # 获取盆骨关节在qpos中的索引(用于提取位置)
        root_jnt_adr = self._client.model.body("pelvis").jntadr[0]
        root_qpos_adr = self._client.model.joint(root_jnt_adr).qposadr[0]
        # 提取盆骨的位置信息（qpos中包含位置和姿态，此处取z坐标）
        qpos = self._client.get_qpos()[root_qpos_adr:root_qpos_adr+7]  # 7DoF（3位置+4四元数）
        # 检查是否自我碰撞（如手臂与躯干碰撞）
        contact_flag = self._client.check_self_collisions()

        # 终止条件：
        terminate_conditions = {
            "qpos[2]_ll": (qpos[2] < 0.9),  # 骨盆高度低于0.9m（可能摔倒）
            "qpos[2]_ul": (qpos[2] > 1.4),  # 骨盆高度高于1.4m（异常状态）
            "contact_flag": contact_flag    # 存在自我碰撞
        }
        # 任何条件满足则任务终止
        return True in terminate_conditions.values()
    
    def reset(self):
        pass


class H1Env(mujoco_env.MujocoEnv):
    def __init__(self, path_to_yaml = None):
        # 加载YAML配置文件（默认使用base.yaml）
        if path_to_yaml is None:
            path_to_yaml = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'configs/base.yaml')
        self.cfg = config_builder.load_yaml(path_to_yaml)

        # 设置仿真时间步和控制频率
        sim_dt = self.cfg.sim_dt  # 物理仿真步长（如0.001s）
        control_dt = self.cfg.control_dt  # 控制决策步长（如0.025s）
        frame_skip = (control_dt/sim_dt)  # 每控制步执行的仿真步数

        # 随机化和扰动的时间间隔
        self.dynrand_interval = self.cfg.dynamics_randomization.interval/control_dt
        self.perturb_interval = self.cfg.perturbation.interval/control_dt
        self.history_len = self.cfg.obs_history_len  # 观测历史长度

        # 生成或加载MuJoCo XML模型（简化版H1机器人）
        path_to_xml = '/tmp/mjcf-export/h1/h1.xml'

        if not os.path.exists(path_to_xml):
            export_dir = os.path.dirname(path_to_xml)

            builder(export_dir, config={
                'unused_joints': [WAIST_JOINTS, ARM_JOINTS],  # 移除腰部和手臂关节
                'rangefinder': False,
                'raisedplatform': False,
                'ctrllimited': self.cfg.ctrllimited,
                'jointlimited': self.cfg.jointlimited,
                'minimal': self.cfg.reduced_xml,  # 是否使用简化模型
            })

        # 关键：确认文件存在并打印路径
        if os.path.exists(path_to_xml):
            print(f"加载模型: {path_to_xml}")
        else:
            print(f"错误: 文件 {path_to_xml} 不存在！")
            raise FileNotFoundError(f"模型文件不存在: {path_to_xml}")     

        # 进行模型的调试检查
        self.model = load_mujoco_model_test(path_to_xml)
        
        # 初始化MuJoCo环境
        super().__init__(self.model, sim_dt, control_dt)

        # 获取所有body的名称（适配新版本API）
        body_names = []
        for i in range(self.model.nbody):  # self.model.nbody是body的总数
            # 通过mj_name2id获取第i个body的名称
            body_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, i)
            body_names.append(body_name)
        print("所有可用的body名称：", body_names)

        # 手动调整机器人质量（匹配实际物理参数）
        self.model.body("pelvis").mass = 8.89
        self.model.body("torso_link").mass = 21.289


        # 设置机器人接口和任务
        self.leg_names = LEG_JOINTS # 腿部关节名称列表
        gains_dict = self.cfg.pdgains.to_dict()
        kp, kd = zip(*[gains_dict[jn+"_joint"] for jn in self.leg_names])
        pdgains = np.array([kp, kd])  # 从配置获取PD控制器增益

        # 定义初始姿态（半坐姿）
        base_position = [0, 0, 0.98]     # 根位置（高度0.98m）
        base_orientation = [1, 0, 0, 0]  # 根姿态（四元数）
        half_sitting_pose = [
            0, 0, -0.2, 0.6, -0.4,
            0, 0, -0.2, 0.6, -0.4,
        ]                                # 左右腿关节角度

        # 初始化任务（定义奖励和终止条件）
        self.interface = robot_interface.RobotInterface(self.model, self.data, 'right_ankle_link', 'left_ankle_link', None)
        self.nominal_pose = base_position + base_orientation + half_sitting_pose

        # 初始化机器人基类（整合PD控制和任务逻辑）
        self.task = Task(self.interface, half_sitting_pose)

        self.robot = RobotBase(pdgains, control_dt, self.interface, self.task)

        # 设置动作空间（仅腿部关节）
        action_space_size = len(self.leg_names)
        action = np.zeros(action_space_size)
        self.action_space = np.zeros(action_space_size)
        self.prev_prediction = np.zeros(action_space_size) # 用于动作平滑

        # 设置观测空间（包含历史状态）e
        self.base_obs_len = 35   # 单个时间步的观测维度
        self.observation_history = collections.deque(maxlen=self.history_len)
        self.observation_space = np.zeros(self.base_obs_len*self.history_len)

        # 定义观测归一化参数（均值和标准差
        self.obs_mean = np.concatenate((
            np.zeros(5),
            half_sitting_pose, np.zeros(10), np.zeros(10),  # 关节位置、速度、力矩
        ))

        self.obs_std = np.concatenate((
            [0.2, 0.2, 1, 1, 1],           # 根姿态和角速度的缩放
            0.5*np.ones(10), 4*np.ones(10), 100*np.ones(10),   # 关节状态的缩放
        ))

        self.obs_mean = np.tile(self.obs_mean, self.history_len)  # 扩展到历史长度
        self.obs_std = np.tile(self.obs_std, self.history_len)

        # copy the original model
        self.default_model = copy.deepcopy(self.model)

    def get_obs(self):
        # 获取机器人状态
        qpos = np.copy(self.interface.get_qpos())  # 广义坐标
        qvel = np.copy(self.interface.get_qvel())  # 广义速度

        # 将标量转换为1维数组（使用np.array包装）
        root_euler = tf3.euler.quat2euler(qpos[3:7])  # 完整的欧拉角（3个分量）
        root_r = np.array([root_euler[0]])  # 横滚角 → 形状 (1,)
        root_p = np.array([root_euler[1]])  # 俯仰角 → 形状 (1,)

        root_ang_vel = qvel[3:6]  # 根节点角速度
        motor_pos = self.interface.get_act_joint_positions()  # 关节位置 → 形状 (N,) 
        motor_vel = self.interface.get_act_joint_velocities()  # 关节速度 → 形状 (N,)
        motor_tau = self.interface.get_act_joint_torques()  # 关节力矩 → 形状 (N,)
        
        # 添加观测噪声（模拟传感器误差）
        if self.cfg.observation_noise.enabled:
            noise_type = self.cfg.observation_noise.type
            scales = self.cfg.observation_noise.scales
            level = self.cfg.observation_noise.multiplier
            # 根据配置选择噪声类型（均匀或高斯）
            noise = lambda x, n : np.random.uniform(-x, x, n) if noise_type=="uniform" else np.random.randn(n) * x
            # 对各观测项添加噪声 噪声维度需与变量匹配（此时root_r是1维数组，长度为1）
            root_r += noise(scales.root_orient * level, len(root_r))  # len(root_r) = 1
            root_p += noise(scales.root_orient * level, len(root_p))  
            root_ang_vel += noise(scales.root_ang_vel * level, len(root_ang_vel))
            motor_pos += noise(scales.motor_pos * level, len(motor_pos))
            motor_vel += noise(scales.motor_vel * level, len(motor_vel))
            motor_tau += noise(scales.motor_tau * level, len(motor_tau))
            
        
        # 拼接观测向量（所有输入均为1维数组）
        robot_state = np.concatenate([root_r, root_p, root_ang_vel, motor_pos, motor_vel, motor_tau])
        
        # 检查维度一致性
        assert robot_state.shape==(self.base_obs_len,), \
            f"State vector length expected to be: {self.base_obs_len} but is {len(robot_state)}"
        
        # 更新观测历史
        if len(self.observation_history)==0:
            for _ in range(self.history_len):
                self.observation_history.appendleft(np.zeros_like(robot_state))
        self.observation_history.appendleft(robot_state)
        
        return np.array(self.observation_history).flatten()  # 返回扁平化的历史观测


    def step(self,action):
        # 动作平滑(减少突变)
        targets = self.cfg.action_smoothing * action + \
            (1 - self.cfg.action_smoothing) * self.prev_prediction
        
        # 计算关节角度偏移(基于半坐姿)
        offsets = [
            self.nominal_pose[self.interface.get_jnt_qposadr_by_name(jnt)[0]]
            for jnt in self.leg_names
        ]

        # 执行控制步(调用RobotBase.step)
        rewards, done = self.robot.step(targets,np.asarray(offsets))
        obs = self.get_obs() #获取当前观测

        # 动态随机化(模拟不同物理环境)
        if self.cfg.dynamics_randomization.enable and np.random.randint(self.dynrand_interval)==0:
            self.randomize_dyn()

        # 施加随机扰动（测试鲁棒性）
        if self.cfg.perturbation.enable and np.random.randint(self.perturb_interval)==0:
            self.randomize_perturb()

        self.prev_prediction = action  # 保存当前动作用于下一次平滑

        return obs, sum(rewards.values()), done, rewards  # 返回观测、总奖励、终止标志和奖励字典


    def reset_model(self):
        # 动态随机化（可选）
        if self.cfg.dynamics_randomization.enable:
            self.randomize_dyn()

        # 初始化状态（基于半坐姿）
        init_qpos, init_qvel = self.nominal_pose.copy(), [0] * self.interface.nv()

        # 添加初始化噪声（随机扰动初始姿态）
        c = self.cfg.init_noise * np.pi/180  # 角度噪声（弧度）
        root_adr = self.interface.get_jnt_qposadr_by_name('root')[0]
        init_qpos[root_adr+2] = np.random.uniform(1.0, 1.02)  # 随机化根高度
        init_qpos[root_adr+3:root_adr+7] = tf3.euler.euler2quat(  # 随机化根姿态
            np.random.uniform(-c, c), np.random.uniform(-c, c), 0)
        init_qpos[root_adr+7:] += np.random.uniform(-c, c, len(self.leg_names))  # 随机化关节角度

        # 设置初始状态并执行几步仿真（稳定物理环境）
        self.set_state(np.asarray(init_qpos), np.asarray(init_qvel))
        for _ in range(3):
            self.interface.step()

        # 重置任务和观测历史
        self.task.reset()
        self.prev_prediction = np.zeros_like(self.prev_prediction)
        self.observation_history = collections.deque(maxlen=self.history_len)
        obs = self.get_obs()

        return obs

    def randomize_perturb(self):
        # 随机施加外部力和力矩（测试鲁棒性）
        frc_mag = self.cfg.perturbation.force_magnitude
        tau_mag = self.cfg.perturbation.force_magnitude
        for body in self.cfg.perturbation.bodies:
            self.data.body(body).xfrc_applied[:3] = np.random.uniform(-frc_mag, frc_mag, 3)  # 力
            self.data.body(body).xfrc_applied[3:] = np.random.uniform(-tau_mag, tau_mag, 3)  # 力矩
            if np.random.randint(2)==0:  # 50%概率不施加扰动
                self.data.xfrc_applied = np.zeros_like(self.data.xfrc_applied)

    def randomize_dyn(self):
        # 随机化动力学参数（质量、摩擦、阻尼等）
        dofadr = [self.interface.get_jnt_qveladr_by_name(jn) for jn in self.leg_names]
        for jnt in dofadr:
            self.model.dof_frictionloss[jnt] = np.random.uniform(0, 2)  # 关节摩擦
            self.model.dof_damping[jnt] = np.random.uniform(0.02, 2)  # 关节阻尼

        # 随机化质心位置
        bodies = ["pelvis"] + [self.model.body(self.model.joint(jn).bodyid).name for jn in self.leg_names]
        for body in bodies:
            default_mass = self.default_model.body(body).mass[0]
            default_ipos = self.default_model.body(body).ipos
            self.model.body(body).mass[0] = default_mass*np.random.uniform(0.95, 1.05)  # 质量±5%
            self.model.body(body).ipos = default_ipos + np.random.uniform(-0.01, 0.01, 3)  # 质心位置扰动


    def viewer_setup(self):
        super().viewer_setup()
        self.viewer.cam.distance = 5  # 相机距离
        self.viewer.cam.lookat[2] = 1.5  # 相机注视点高度
        self.viewer.cam.lookat[0] = 1.0  # 相机注视点x坐标


def load_mujoco_model_test(xml_path, verbose=True):
    """
    加载MuJoCo模型并进行调试检查
    
    参数:
    - xml_path: XML文件路径
    - verbose: 是否打印详细调试信息
    
    返回:
    - mujoco.MjModel: 加载成功的MuJoCo模型
    """
    if verbose:
        print(f"Path to XML: {xml_path}")
        print(f"os.path.exists(xml_path): {os.path.exists(xml_path)}")
    
    # 检查文件是否存在
    if not os.path.exists(xml_path):
        raise FileNotFoundError(f"模型文件不存在: {xml_path}")
    
    try:
        # 读取XML内容用于调试
        with open(xml_path, 'r') as f:
            xml_content = f.read()
            if verbose:
                print(f"XML内容检查 - 包含'pelvis': {'pelvis' in xml_content}")
        
        # 加载模型
        if verbose:
            print("尝试加载模型...")
        model = mujoco.MjModel.from_xml_path(xml_path)
        if verbose:
            print("模型加载成功")
        
        # 打印模型基本信息
        if verbose:
            print(f"模型body数量: {model.nbody}")
            print(f"模型joint数量: {model.njnt}")
            
            # 打印所有body名称
            body_names = [model.body(i).name for i in range(model.nbody)]
            print("加载后的所有body名称:", body_names)
        
        return model
    
    except Exception as e:
        if verbose:
            print(f"加载模型失败: {e}")
            import traceback
            traceback.print_exc()
        raise
    
    # 检查特定body是否存在
    def check_body_exists(model, body_name, verbose=True):
        """检查模型中是否存在指定名称的body"""
        try:
            body_id = model.body_name2id(body_name)
            if verbose:
                print(f"找到{body_name}，ID: {body_id}")
            return True
        except Exception as e:
            if verbose:
                print(f"未找到{body_name}: {e}")
                print("可用的body名称:", [model.body(i).name for i in range(model.nbody)])
            return False