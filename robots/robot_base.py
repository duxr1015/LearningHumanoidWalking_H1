import numpy as np

class RobotBase(object):
    def __init__(self, pdgains, dt, client, task, pdrand_k=0,sim_bemf = False, sim_motor_dyn = False):

        self.client = client  # 机器人接口
        self.task = task      # 任务逻辑
        self.control_dt = dt  # 控制时间步长
        self.pdrand_k = pdrand_k # PD参数随机化系数,PD 增益随机化系数（如 0.1 表示 ±10% 随机波动），增强策略鲁棒性
        self.sim_bemf = sim_bemf # 是否模拟反电动势（与关节速度成正比的阻力），增加物理真实性
        self.sim_motor_dyn = sim_motor_dyn  #是否模拟电机动力学,模拟电机动态响应（如延迟、饱和），更精确地建模电机行为

        #禁止同时模拟反电动势和电机动力学(计算冲突)
        assert(self.sim_bemf & self.sim_motor_dyn == False), \
            "You cannot simulate back-EMF and motor dynamics simultaneously!"
        
        # 设置PD增益（按关节配置）
        self.kp = pdgains[0]  # 比例增益
        self.kd = pdgains[1]  # 微分增益
        assert self.kp.shape==self.kd.shape==(self.client.nu(),), \
            f"kp shape {self.kp.shape} and kd shape {self.kd.shape} must be {(self.client.nu(),)}"
        
        # 反电动势阻尼系数（用于模拟电机阻力）
        self.tau_d = np.zeros(self.client.nu())

        # 初始化PD控制器并验证
        self.client.set_pd_gains(self.kp, self.kd)
        tau = self.client.step_pd(np.zeros(self.client.nu()), np.zeros(self.client.nu()))
        w = self.client.get_act_joint_velocities()
        assert len(w)==len(tau)

        # 记录历史动作和力矩（用于奖励计算）
        self.prev_action = None
        self.prev_torque = None
        self.iteration_count = np.inf

        # 计算帧跳过次数（控制步长/仿真步长）
        if (np.around(self.control_dt%self.client.sim_dt(), 6)):
            raise Exception("Control dt should be an integer multiple of Simulation dt.")
        self.frame_skip = int(self.control_dt/self.client.sim_dt())

    def _do_simulation(self, target, n_frames):
        # 随机化PD增益（增强鲁棒性）
        if self.pdrand_k:
            k = self.pdrand_k
            kp = np.random.uniform((1-k)*self.kp, (1+k)*self.kp)
            kd = np.random.uniform((1-k)*self.kd, (1+k)*self.kd)
            self.client.set_pd_gains(kp, kd)

        assert target.shape == (self.client.nu(),), \
            f"Target shape must be {(self.client.nu(),)}"

        ratio = self.client.get_gear_rations()  # 获取齿轮比（转换电机力矩为关节力矩）

        # 随机更新反电动势阻尼系数（仅当sim_bemf=True且10%概率时）
        if self.sim_bemf and np.random.randint(10)==0:
            self.tau_d = np.random.uniform(5, 40, self.client.nu())

        # 执行多次仿真步（frame_skip次）
        for _ in range(n_frames):
            w = self.client.get_act_joint_velocities()  # 获取当前关节速度
            tau = self.client.step_pd(target, np.zeros(self.client.nu()))  # PD控制器计算力矩

            # 应用反电动势（速度相关的阻力）
            if self.sim_bemf:
                tau = tau - self.tau_d*w

            # 考虑齿轮比（电机力矩→关节力矩的转换）
            tau /= ratio

            # 设置电机力矩并推进仿真
            self.client.set_motor_torque(tau, self.sim_motor_dyn)
            self.client.step()

    def step(self, action, offset=None):
        # 类型和维度检查
        if not isinstance(action, np.ndarray):
            raise TypeError("Expected action to be a numpy array")
        action = np.copy(action)
        assert action.shape == (self.client.nu(),), \
            f"Action vector length expected to be: {self.client.nu()} but is {action.shape}"
        
        # 应用位置偏移（如基于中立姿态的相对位置）
        if offset is not None:
            if not isinstance(offset, np.ndarray):
                raise TypeError("Expected offset to be a numpy array")
            assert offset.shape == action.shape, \
                f"Offset shape {offset} must match action shape {action.shape}"
            action += offset
        
        # 记录历史动作和力矩（用于奖励计算）
        if self.prev_action is None:
            self.prev_action = action
        if self.prev_torque is None:
            self.prev_torque = np.asarray(self.client.get_act_joint_torques())
        
        # 执行仿真
        self._do_simulation(action, self.frame_skip)
        
        # 任务相关操作（计算奖励、检查终止条件）
        self.task.step()
        rewards = self.task.calc_reward(self.prev_torque, self.prev_action, action)
        done = self.task.done()
        
        # 更新历史记录
        self.prev_action = action
        self.prev_torque = np.asarray(self.client.get_act_joint_torques())
        
        return rewards, done

    

    

